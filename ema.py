import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import copy
import math
import sys
import argparse
import time
import datetime
from util import dist
from typing import Iterable
from functools import reduce
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_videoqa_dataset, videoqa_collate_fn
from model import build_model, get_tokenizer
from args import get_args_parser
from util.misc import get_mask, adjust_learning_rate
from util.metrics import MetricLogger
from kd_loss import ListwiseLoss, PairwiseLoss
from metrics.eval_metrics import ndcg_at_k


def update_ema(model, model_ema, decay):
    """Apply exponential moving average update.

    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        if hasattr(model, "module"):
            # unwrapping DDP
            model = model.module
        msd = model.state_dict()
        for k, ema_v in model_ema.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)
            

def train_one_epoch(
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        model_ema: torch.nn.Module,
        tokenizer,
        listwise_loss,
        pairwise_loss,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        dataset_name,
        args,
        max_norm: float = 0,
):
    student.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)

    for i_batch, batch_dict in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        text_ids = encoded["input_ids"].to(device)
        text_mask = encoded["attention_mask"].to(device)
        mask = text_ids == tokenizer.mask_token_id
        delay = args.max_feats if args.use_video else 0

        # teacher forward
        with torch.no_grad():
            teacher_output = teacher(
                video=video,
                video_mask=video_mask,
                input_ids=text_ids,
                attention_mask=text_mask
            )
            t_logits = teacher_output["logits"][:, delay:][mask]  # (B, C)
            t_states = teacher_output["adapter_states"][-1]
            t_reps = teacher_output["last_hidden_state"][:, delay:][mask]

        # forward
        student_output = student(
            video=video,
            video_mask=video_mask,
            input_ids=text_ids,
            attention_mask=text_mask
        )
        s_logits = student_output["logits"][:, delay:][mask]
        s_states = student_output["adapter_states"][-1]

        answer_id = batch_dict["answer_id"].to(device)

        loss = 0
        if dataset_name == "ivqa":
            a = (answer_id / 2).clamp(max=1)
            nll = -F.log_softmax(s_logits, 1, _stacklevel=5)
            c_loss = (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean()
        else:
            c_loss = F.cross_entropy(s_logits, answer_id)

        loss += c_loss
        display_dict = {"cls_loss": c_loss}

        # listwise loss
        if args.alpha_l > 0:
            l_loss = listwise_loss(t_logits, s_logits)
            loss += args.alpha_l * l_loss
            display_dict.update({"list_loss": l_loss})

        # pairwise loss
        if args.alpha_p > 0:
            sample_logits = []
            k = random.randint(2, args.num_sample)
            for _ in range(k):
                sample_logits.append(teacher.predict(t_reps, enable_dropout=True))
            sample_logits = torch.stack(sample_logits, dim=1)
            p_loss = pairwise_loss(answer_id, t_logits, s_logits, sample_logits)
            loss += args.alpha_p * p_loss
            display_dict.update({"pair_loss": p_loss})

        # feature loss
        if args.alpha_f > 0:
            f_loss = F.mse_loss(t_states, s_states)
            loss += args.alpha_f * f_loss
            display_dict.update({"feat_loss": f_loss})

        # vanilla kd loss
        if args.alpha_v > 0:
            v_loss = F.kl_div(torch.log_softmax(s_logits / args.temperature, dim=1),
                              torch.softmax(t_logits / args.temperature, dim=1),
                              reduction="batchmean") * args.temperature ** 2
            loss += args.alpha_v * v_loss
            display_dict.update({"vanilla_kd_loss": v_loss})

        display_dict.update({"total_loss": loss})

        # reduce losses over all GPUs for logging purposes
        display_dict_reduced = dist.reduce_dict(display_dict)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            curr_step=epoch * len(data_loader) + i_batch,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(student, model_ema, 0.9998)

        metric_logger.update(**display_dict_reduced)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        tokenizer,
        data_loader,
        device: torch.device,
        dataset_name,
        args,
        thresholds=[1, 10],
        split="test",
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    res = {}

    for i_batch, batch_dict in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if (
                not args.suffix and not args.use_context
        ):  # remove sep token if not using the suffix
            attention_mask[input_ids == tokenizer.sep_token_id] = 0
            input_ids[input_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id

        output = model(
            video=video,
            video_mask=video_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        mask = input_ids == tokenizer.mask_token_id
        delay = args.max_feats if args.use_video else 0

        logits = output["logits"][:, delay:][mask]

        logits = logits.softmax(-1)
        topk_aids = torch.topk(logits, max(thresholds), -1).indices

        answer_id, qids = batch_dict["answer_id"].to(device), batch_dict["qid"]
        batch_ndcg = ndcg_at_k(logits, answer_id, k=5)
        if dataset_name == "ivqa":
            answer_id = (answer_id / 2).clamp(max=1)
            answer_id_expanded = answer_id.to(device)
        else:
            answer_id_expanded = answer_id.view(-1, 1).expand_as(topk_aids).to(device)

        agreeings = {}
        for x in thresholds:
            if dataset_name == "ivqa":
                predicted = F.one_hot(
                    topk_aids[:, :x], num_classes=answer_id_expanded.shape[-1]
                ).sum(1)
                agreeings[x] = (predicted * answer_id_expanded).max(1)[0]
            else:
                agreeings[x] = topk_aids[:, :x] == answer_id_expanded[:, :x]

        for i, (qid, gt, pred, ndcg) in enumerate(
                zip(qids, answer_id, topk_aids, batch_ndcg)
        ):
            res[qid] = {
                "pred": pred.tolist(),
                "gt": gt.tolist() if dataset_name == "ivqa" else gt.item(),
                "ndcg": ndcg.item()
            }
            for x in thresholds:
                res[qid][f"acc{x}"] = agreeings[x][i].sum().detach().cpu().item()

        dico = {"acc": agreeings[1].mean(), "ndcg": batch_ndcg.mean()}
        dico_reduced = dist.reduce_dict(dico)
        acc_value = dico_reduced["acc"].item()
        metric_logger.update(acc=dico_reduced["acc"].item(), ndcg=dico_reduced["ndcg"].item())

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    out = {}
    for x in thresholds:
        out[f"acc{x}"] = sum(results[qid][f"acc{x}"] for qid in results) / len(results)
    out["ndcg"] = sum(results[qid]["ndcg"] for qid in results) / len(results)

    if dist.is_main_process():
        print(dataset_name)
        for x in thresholds:
            print(f"acc{x}: {out[f'acc{x}']: .2%}")
        print(f"ndcg@5: {out['ndcg']: .2%}")
    return results, out


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if dist.is_main_process():
        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        print(args)

    device = torch.device(args.device)

    # Fix seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build dataset
    tokenizer = get_tokenizer(args)

    dataset_test = build_videoqa_dataset(
        args.dataset,
        "test",
        args,
        tokenizer,
    )
    sampler_test = (
        DistributedSampler(dataset_test, shuffle=False)
        if args.distributed
        else torch.utils.data.SequentialSampler(dataset_test)
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size_val,
        sampler=sampler_test,
        collate_fn=videoqa_collate_fn,
        num_workers=args.num_workers,
    )

    if not args.eval:
        dataset_train = build_videoqa_dataset(args.dataset, "train", args, tokenizer)
        sampler_train = (
            DistributedSampler(dataset_train)
            if args.distributed
            else torch.utils.data.RandomSampler(dataset_train)
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=sampler_train,
            collate_fn=videoqa_collate_fn,
            num_workers=args.num_workers,
        )
    else:
        dataloader_train = None

    args.n_ans = len(dataloader_test.dataset.a2id)
    model = build_model(args)
    model.to(device)

    # setting loss
    listwise_loss = ListwiseLoss(args)
    pairwise_loss = PairwiseLoss(args)

    # load teacher
    teacher = None
    if args.teacher_load and not args.eval:
        if dist.is_main_process():
            print("loading teacher from %s" % args.teacher_load)
        teacher = copy.deepcopy(model)
        checkpoint = torch.load(args.teacher_load, map_location="cpu")
        teacher.load_state_dict(checkpoint["model"], strict=False)
        teacher.eval()
        teacher.lm_predictions.lm_head.dropout.train()
        model.lm_predictions.lm_head.dropout = teacher.lm_predictions.lm_head.dropout

    # Set up optimizer
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        params_for_optimization,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    # load student
    if args.load:
        if dist.is_main_process():
            print("loading student from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    aid2tokid = torch.zeros(len(dataloader_test.dataset.a2id), args.max_atokens).long()
    for a, aid in dataloader_test.dataset.a2id.items():
        tok = torch.tensor(
            tokenizer(
                a,
                add_special_tokens=False,
                max_length=args.max_atokens,
                truncation=True,
                padding="max_length",
            )["input_ids"],
            dtype=torch.long,
        )
        aid2tokid[aid] = tok
    model.set_answer_embeddings(
        aid2tokid.to(model.device), freeze_last=args.freeze_last
    )  # init answer embedding module
    model_ema = copy.deepcopy(model)

    if not args.eval:
        teacher.set_answer_embeddings(aid2tokid.to(model.device))
        if dist.is_main_process():
            print("Start training")
        start_time = time.time()
        best_epoch = args.start_epoch
        best_acc = 0
        for epoch in range(args.start_epoch, args.epochs):
            if dist.is_main_process():
                print(f"Starting epoch {epoch}")
            if args.distributed:
                sampler_train.set_epoch(epoch)
            train_stats = train_one_epoch(
                teacher=teacher,
                student=model,
                model_ema=model_ema,
                tokenizer=tokenizer,
                listwise_loss=listwise_loss,
                pairwise_loss=pairwise_loss,
                data_loader=dataloader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                dataset_name=args.dataset,
                args=args,
                max_norm=args.clip_max_norm,
            )

            if (epoch + 1) % args.eval_skip == 0:
                val_stats = {}
                print(f"Validating {args.dataset}")

                curr_val_stats, out = evaluate(
                    model=model_ema,
                    tokenizer=tokenizer,
                    data_loader=dataloader_test,
                    device=device,
                    dataset_name=args.dataset,
                    args=args,
                    split="val"
                )
                val_stats.update(
                    {args.dataset + "_" + k: v for k, v in out.items()}
                )
                if out["acc1"] > best_acc:
                    best_epoch = epoch
                    best_acc = out["acc1"]

                    if dist.is_main_process() and args.save_dir:
                        checkpoint_path = os.path.join(
                            args.save_dir, f"best_model.pth"
                        )
                        dist.save_on_master(
                            {
                                "model": model_ema.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "args": args,
                            },
                            checkpoint_path,
                        )
            else:
                val_stats = {}

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
            }

            if args.save_dir and dist.is_main_process():
                with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
        # load best ckpt
        if dist.is_main_process() and args.save_dir:
            print(f"loading best checkpoint from epoch {best_epoch}")
        if args.save_dir:
            if args.distributed:
                torch.distributed.barrier()  # wait all processes
            checkpoint = torch.load(
                os.path.join(args.save_dir, f"best_model.pth"),
                map_location="cpu",
            )
            model.load_state_dict(checkpoint["model"], strict=False)

    evaluate(
        model=model,
        tokenizer=tokenizer,
        data_loader=dataloader_test,
        device=device,
        dataset_name=args.dataset,
        args=args,
        split="test"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    main(args)
