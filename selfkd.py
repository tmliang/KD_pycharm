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
import logging
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
from kd_loss import build_ranking_loss
from metrics.ndcg import ndcg_at_k
from metrics.eval_metrics import mean_average_precision

logging.basicConfig(level=logging.ERROR)

M = 10

def train_one_epoch(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    tokenizer,
    ranking_loss,
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

        # teacher forward
        with torch.no_grad():
            teacher_output = teacher(
                video=video,
                video_mask=video_mask,
                input_ids=text_ids,
                attention_mask=text_mask
            )

        # forward
        student_output = student(
            video=video,
            video_mask=video_mask,
            input_ids=text_ids,
            attention_mask=text_mask
        )

        mask = encoded["input_ids"] == tokenizer.mask_token_id
        delay = args.max_feats if args.use_video else 0
        t_logits = teacher_output["logits"][:, delay: text_ids.size(1) + delay][mask]  # (B, C)
        s_logits = student_output["logits"][:, delay: text_ids.size(1) + delay][mask]

        answer_id = batch_dict["answer_id"].to(device)
        rank_loss = ranking_loss(answer_id, t_logits, s_logits)
        ndcg = ndcg_at_k(answer_id, t_logits, s_logits, k=10)

        if dataset_name == "ivqa":
            a = (answer_id / 2).clamp(max=1)
            nll = -F.log_softmax(s_logits, 1, _stacklevel=5)
            cls_loss = (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean()
        else:
            cls_loss = F.cross_entropy(s_logits, answer_id)

        loss = cls_loss + args.alpha * rank_loss
        loss_dict = {"loss": loss, "cls_loss": cls_loss, "rank_loss": rank_loss, "ndcg": ndcg}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)


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

        metric_logger.update(**loss_dict_reduced)
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

        logits = output["logits"]
        delay = args.max_feats if args.use_video else 0
        logits = logits[:, delay : encoded["input_ids"].size(1) + delay][
            encoded["input_ids"] == tokenizer.mask_token_id
        ]  # get the prediction on the mask token
        logits = logits.softmax(-1)
        topk_aids = torch.topk(logits, max(thresholds), -1).indices

        answer_id, qids = batch_dict["answer_id"].to(device), batch_dict["qid"]
        types = batch_dict["type"]
        if "sub" in batch_dict:
            subs = batch_dict["sub"]
        else:
            subs = [0] * len(types)
        if dataset_name == "ivqa":
            answer_id = (answer_id / 2).clamp(max=1)
            answer_id_expanded = answer_id.to(device)
        else:
            answer_id_expanded = answer_id.view(-1, 1).expand_as(topk_aids).to(device)

        maps = mean_average_precision(topk_aids, answer_id)

        agreeings = {}
        for x in thresholds:
            if dataset_name == "ivqa":
                predicted = F.one_hot(
                    topk_aids[:, :x], num_classes=answer_id_expanded.shape[-1]
                ).sum(1)
                agreeings[x] = (predicted * answer_id_expanded).max(1)[0]
            else:
                agreeings[x] = topk_aids[:, :x] == answer_id_expanded[:, :x]

        for i, (qid, gt, pred) in enumerate(
            zip(qids, answer_id, topk_aids)
        ):
            res[qid] = {
                "pred": pred.tolist(),
                "gt": gt.tolist() if dataset_name == "ivqa" else gt.item(),
                "map": maps.item()
            }
            for x in thresholds:
                res[qid][f"acc{x}"] = agreeings[x][i].sum().detach().cpu().item()

        dico = {"acc": agreeings[1].sum() / len(qids)}
        dico_reduced = dist.reduce_dict(dico)
        acc_value = dico_reduced["acc"].item()
        metric_logger.update(acc=acc_value)

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    out = {}
    for x in thresholds:
        out[f"acc{x}"] = sum(results[qid][f"acc{x}"] for qid in results) / len(results)
    out["map"] = sum(results[qid][f"map"] for qid in results) / len(results)

    if dist.is_main_process():
        print(dataset_name)
        for x in thresholds:
            print(f"acc{x}: {out[f'acc{x}']: .2%}")
        print(f"map@10: {out['map']: .2%}")
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
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.is_main_process():
        print("number of params:", n_parameters)

    # Set up optimizer
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(
        params_for_optimization,
        lr=args.lr
    )

    # load student
    if args.load:
        if dist.is_main_process():
            print("loading student from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    # setting loss
    ranking_loss = build_ranking_loss(args)

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
    if not args.eval:
        if dist.is_main_process():
            print("Start training")

        best_acc = 0
        teacher = copy.deepcopy(model)
        for m in range(M):
            checkpoint_path = os.path.join(args.save_dir, f"best_model.pth")
            if m == 0:
                if dist.is_main_process():
                    print("loading teacher from %s" % args.teacher_load)
                teacher.load_state_dict(torch.load(args.teacher_load, map_location="cpu")["model"], strict=False)
            else:
                if dist.is_main_process():
                    print(f"updating teacher")
                teacher.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"], strict=False)

            teacher.eval()
            teacher.set_answer_embeddings(aid2tokid.to(model.device))
            print(f"Starting iteration {m}")
            for epoch in range(0, args.epochs):
                if dist.is_main_process():
                    print(f"Starting epoch {epoch}")
                if args.distributed:
                    sampler_train.set_epoch(epoch)
                train_stats = train_one_epoch(
                    teacher=teacher,
                    student=model,
                    tokenizer=tokenizer,
                    ranking_loss=ranking_loss,
                    data_loader=dataloader_train,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    dataset_name=args.dataset,
                    args=args,
                    max_norm=args.clip_max_norm,
                )

                val_stats = {}
                print(f"Validating {args.dataset}")

                curr_val_stats, out = evaluate(
                    model=model,
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
                    best_acc = out["acc1"]
                    if dist.is_main_process() and args.save_dir:
                        print(f"updating best acc: {best_acc}")
                        dist.save_on_master({"model": model.state_dict()}, checkpoint_path)

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

                if args.save_dir and dist.is_main_process():
                    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

            print(f"best acc until now: {best_acc}")
            if args.distributed:
                torch.distributed.barrier()  # wait all processes

    if args.save_dir:
        if args.distributed:
            torch.distributed.barrier()  # wait all processes
        checkpoint = torch.load(
            os.path.join(args.save_dir, f"best_model.pth"),
            map_location="cpu",
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    results, out = evaluate(
        model=model,
        tokenizer=tokenizer,
        data_loader=dataloader_test,
        device=device,
        dataset_name=args.dataset,
        args=args,
        split="test"
    )

    if args.save_dir and dist.is_main_process():
        json.dump(
            results,
            open(os.path.join(args.save_dir, args.dataset + ".json"), "w"),
        )
        json.dump(
            out,
            open(
                os.path.join(args.save_dir, args.dataset + "summary.json"), "w"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    main(args)
