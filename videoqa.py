import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
from typing import Iterable
import argparse
import time
import datetime
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from functools import reduce

from datasets import build_videoqa_dataset, videoqa_collate_fn
from model import build_model, get_tokenizer
from args import get_args_parser
from util.misc import get_mask, adjust_learning_rate
from util.metrics import MetricLogger
from metrics.eval_metrics import mean_average_precision


def train_one_epoch(
    model: torch.nn.Module,
    tokenizer,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    dataset_name,
    args,
    max_norm: float = 0,
):
    model.train()
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

        # forward
        output = model(
            video=video,
            video_mask=video_mask,
            input_ids=text_ids,
            attention_mask=text_mask
        )

        delay = args.max_feats if args.use_video else 0
        point_logits = output["logits"][:, delay: text_ids.size(1) + delay][text_ids == tokenizer.mask_token_id]
        rep = output["hidden_states"][:, delay: text_ids.size(1) + delay][text_ids == tokenizer.mask_token_id]

        hib_ins = model.sample_ins_embeddings(rep, args.n_sample)
        hib_ans = model.sample_ans_embeddings(args.n_sample)
        hib_logits = torch.matmul(hib_ins.unsqueeze(1), hib_ans.transpose(1, 2)).mean((2, 3))
        # (B, k, d), (C, k, d) -> (B, C, k, k) -> (B, C)

        answer_id = batch_dict["answer_id"].to(device)

        losses = []
        for l in [point_logits, hib_logits]:
            if dataset_name == "ivqa":
                a = (answer_id / 2).clamp(max=1)
                nll = -F.log_softmax(l, 1, _stacklevel=5)
                losses.append((nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean())
            else:
                losses.append(F.cross_entropy(l, answer_id))

        loss = losses[0] + args.sigma * losses[1]
        loss_dict = {"point_loss": losses[0], "hedge_loss": losses[1]}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
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
    type_map={0: "all"},
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
        text_ids = encoded["input_ids"].to(device)
        text_mask = encoded["attention_mask"].to(device)
        if (
            not args.suffix and not args.use_context
        ):  # remove sep token if not using the suffix
            text_mask[text_ids == tokenizer.sep_token_id] = 0
            text_ids[text_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id

        output = model(
            video=video,
            video_mask=video_mask,
            input_ids=text_ids,
            attention_mask=text_mask,
        )

        delay = args.max_feats if args.use_video else 0
        point_logits = output["logits"][:, delay: text_ids.size(1) + delay][text_ids == tokenizer.mask_token_id]
        rep = output["hidden_states"][:, delay: text_ids.size(1) + delay][text_ids == tokenizer.mask_token_id]
        ins_mu, _ = model.get_ins_distribution(rep)
        ans_mu, _ = model.get_ans_distribution()
        hib_logits = ins_mu @ ans_mu.t()

        point_topk_aids = torch.topk(point_logits, max(thresholds), -1).indices
        hib_topk_aids = torch.topk(hib_logits, max(thresholds), -1).indices

        answer_id, qids = batch_dict["answer_id"].to(device), batch_dict["qid"]
        if dataset_name == "ivqa":
            answer_id = (answer_id / 2).clamp(max=1)
            answer_id_expanded = answer_id.to(device)
        else:
            answer_id_expanded = answer_id.view(-1, 1).expand_as(point_topk_aids).to(device)

        point_maps = mean_average_precision(point_topk_aids, answer_id)
        hib_maps = mean_average_precision(hib_topk_aids, answer_id)

        point_agreeings = {}
        hib_agreeings = {}
        for x in thresholds:
            if dataset_name == "ivqa":
                predicted = F.one_hot(point_topk_aids[:, :x], num_classes=answer_id_expanded.shape[-1]).sum(1)
                point_agreeings[x] = (predicted * answer_id_expanded).max(1)[0]
                predicted = F.one_hot(hib_topk_aids[:, :x], num_classes=answer_id_expanded.shape[-1]).sum(1)
                hib_agreeings[x] = (predicted * answer_id_expanded).max(1)[0]
            else:
                point_agreeings[x] = point_topk_aids[:, :x] == answer_id_expanded[:, :x]
                hib_agreeings[x] = hib_topk_aids[:, :x] == answer_id_expanded[:, :x]

        for i, (qid, gt, point_pred, hib_pred) in enumerate(
                zip(qids, answer_id, point_topk_aids, hib_topk_aids)
        ):
            res[qid] = {
                "point_map": point_maps.item(),
                "hib_map": hib_maps.item(),
            }
            for x in thresholds:
                res[qid][f"acc{x}"] = {"point": point_agreeings[x][i].sum().detach().cpu().item(),
                                       "hib": hib_agreeings[x][i].sum().detach().cpu().item(),
                                       }

        dico = {"point_acc": point_agreeings[1].sum() / len(qids), "hib_acc": hib_agreeings[1].sum() / len(qids)}
        dico_reduced = dist.reduce_dict(dico)
        point_acc = dico_reduced["point_acc"].item()
        hib_acc = dico_reduced["hib_acc"].item()
        metric_logger.update(point_acc=point_acc, hib_acc=hib_acc)

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    out = {}
    for x in thresholds:
        out[f"point_acc{x}"] = sum(results[qid][f"acc{x}"]["point"] for qid in results) / len(results)
        out[f"hib_acc{x}"] = sum(results[qid][f"acc{x}"]["hib"] for qid in results) / len(results)
    out["point_map"] = sum(results[qid][f"point_map"] for qid in results) / len(results)
    out["hib_map"] = sum(results[qid][f"hib_map"] for qid in results) / len(results)

    if dist.is_main_process():
        print(dataset_name)
        for x in thresholds:
            print(f"point_acc{x}: {out[f'point_acc{x}']: .2%}")
            print(f"hib_acc{x}: {out[f'hib_acc{x}']: .2%}")
        print(f"point_map@10: {out['point_map']: .2%}")
        print(f"hib_map@10: {out['hib_map']: .2%}")
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

    # Build model
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

    # Set up optimizer
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(
        params_for_optimization,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    # init answer embedding module
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
    )

    if not args.eval:
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
                model=model,
                tokenizer=tokenizer,
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
                    model=model,
                    tokenizer=tokenizer,
                    data_loader=dataloader_test,
                    device=device,
                    dataset_name=args.dataset,
                    args=args,
                    split="val",
                    type_map=dataloader_test.dataset.type_map,
                )
                val_stats.update(
                    {args.dataset + "_" + k: v for k, v in out.items()}
                )
                if out["point_acc1"] > best_acc:
                    best_epoch = epoch
                    best_acc = out["point_acc1"]

                    if dist.is_main_process() and args.save_dir:
                        checkpoint_path = os.path.join(
                            args.save_dir, f"best_model.pth"
                        )
                        dist.save_on_master(
                            {
                                "model": model.state_dict()
                            },
                            checkpoint_path,
                        )
                        json.dump(
                            curr_val_stats,
                            open(
                                os.path.join(
                                    args.save_dir,
                                    args.dataset + "_val.json",
                                ),
                                "w",
                            ),
                        )
                        json.dump(
                            {"acc": best_acc, "ep": epoch},
                            open(
                                os.path.join(
                                    args.save_dir,
                                    args.dataset + "acc_val.json",
                                ),
                                "w",
                            ),
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
                checkpoint_path = os.path.join(args.save_dir, f"ckpt.pth")
                dist.save_on_master(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

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

    results, out = evaluate(
        model=model,
        tokenizer=tokenizer,
        data_loader=dataloader_test,
        device=device,
        dataset_name=args.dataset,
        args=args,
        split="test",
        type_map=dataloader_test.dataset.type_map,
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
    args = get_args_parser()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    main(args)
