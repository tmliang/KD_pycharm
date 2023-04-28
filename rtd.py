import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import math
import sys
from typing import Iterable
import argparse
import time
import datetime
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from functools import reduce

from datasets import build_videoqa_dataset, videoqa_collate_fn
from model import G_model, D_model, get_tokenizer
from args import get_args_parser
from util.misc import get_mask, adjust_learning_rate
from util.metrics import MetricLogger
from metrics.eval_metrics import mean_average_precision


@torch.no_grad()
def topk_sampling(logits, topk=1, temp=1):
    top_p = torch.nn.functional.softmax(logits / temp, dim=-1)
    topk = max(1, topk)
    next_tokens = torch.multinomial(top_p, topk)
    return next_tokens


@torch.no_grad()
def generate_electra_data(inputs, mask, mask_idx, cand, k, ma):
    b, l = inputs.size()
    mask_idx = mask_idx.float().argmax(1)
    D_inputs = torch.zeros(b * k, l + ma).to(inputs)
    D_mask = torch.ones_like(D_inputs)
    D_mask_labels = torch.zeros_like(D_inputs)
    for i in range(b):
        # make input_ids
        i_inputs = D_inputs[i * k: (i + 1) * k]
        i_inputs[:, :mask_idx[i]] = inputs[i, :mask_idx[i]]
        i_inputs[:, mask_idx[i]: mask_idx[i] + ma] = cand[i]
        i_inputs[:, mask_idx[i] + ma:] = inputs[i, mask_idx[i]:]

        # make attention mask
        i_mask = D_mask[i * k: (i + 1) * k]
        i_mask[:, mask_idx[i]: mask_idx[i] + ma] = cand[i].bool().long()
        i_mask[:, mask_idx[i] + ma:] = mask[i, mask_idx[i]:]

        # make answer mask
        i_labels = D_mask_labels[i * k: (i + 1) * k]
        i_labels[:, mask_idx[i]: mask_idx[i] + ma] = cand[i].bool().long() + 1
    return D_inputs, D_mask, D_mask_labels



def train_one_epoch(
    G: torch.nn.Module,
    D: torch.nn.Module,
    a2tok: torch.tensor,
    tokenizer,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    dataset_name,
    args,
    max_norm: float = 0,
):
    G.train()
    D.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)
    k, ma = args.topk, args.max_atokens

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        answer_id = batch_dict["answer_id"].to(device)
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        G_inputs = encoded["input_ids"].to(device)
        G_mask = encoded["attention_mask"].to(device)

        G_output = G(
            video=video,
            video_mask=video_mask,
            input_ids=G_inputs,
            attention_mask=G_mask
        )
        delay = args.max_feats if args.use_video else 0
        mask_idx = encoded["input_ids"] == tokenizer.mask_token_id
        G_logits = G_output["logits"][:, delay: encoded["input_ids"].size(1) + delay][mask_idx]
        if dataset_name == "ivqa":
            a = (answer_id / 2).clamp(max=1)
            nll = -F.log_softmax(G_logits, 1, _stacklevel=5)
            g_loss = (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean()
        else:
            g_loss = F.cross_entropy(G_logits, answer_id)

        # make electra data
        topk_aids = topk_sampling(G_logits, topk=k)
        D_inputs, D_mask, D_mask_labels = generate_electra_data(G_inputs, G_mask, mask_idx, a2tok[topk_aids], k, ma)
        D_labels = (topk_aids == answer_id.unsqueeze(1)).view(-1).float()

        D_output = D(
            video=video.repeat_interleave(k, dim=0),
            video_mask=video_mask.repeat_interleave(k, dim=0),
            input_ids=D_inputs,
            attention_mask=D_mask,
            mask_labels=D_mask_labels,
            labels=D_labels
        )

        loss = g_loss + args.rtd_lambda * D_output["loss"]
        loss_dict = {"loss": loss, "G_loss": g_loss, "D_loss": D_output["loss"]}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm)
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
    G: torch.nn.Module,
    D: torch.nn.Module,
    a2tok: torch.tensor,
    tokenizer,
    data_loader,
    device: torch.device,
    dataset_name,
    args,
    thresholds=[1, 3, 5],
    split="test",
    type_map={0: "all"},
):
    G.eval()
    D.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    res = {}
    k, ma = args.topk_val, args.max_atokens
    thresholds = [th for th in thresholds if th < k]

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
        G_inputs = encoded["input_ids"].to(device)
        G_mask = encoded["attention_mask"].to(device)
        if (
            not args.suffix and not args.use_context
        ):  # remove sep token if not using the suffix
            G_mask[G_inputs == tokenizer.sep_token_id] = 0
            G_inputs[G_inputs == tokenizer.sep_token_id] = tokenizer.pad_token_id

        G_output = G(
            video=video,
            video_mask=video_mask,
            input_ids=G_inputs,
            attention_mask=G_mask
        )
        delay = args.max_feats if args.use_video else 0
        mask_idx = encoded["input_ids"] == tokenizer.mask_token_id
        G_logits = G_output["logits"][:, delay: encoded["input_ids"].size(1) + delay][mask_idx]

        G_pred = torch.topk(G_logits, k, -1).indices
        D_inputs, D_mask, D_mask_labels = generate_electra_data(G_inputs, G_mask, mask_idx, a2tok[G_pred], k, ma)

        D_output = D(
            video=video.repeat_interleave(k, dim=0),
            video_mask=video_mask.repeat_interleave(k, dim=0),
            input_ids=D_inputs,
            attention_mask=D_mask,
            mask_labels=D_mask_labels
        )
        D_logits = D_output["logits"].view(video.size(0), -1).softmax(-1)
        D_logits, D_pred = D_logits.sort(1, descending=True)
        D_pred = G_pred.gather(1, D_pred)

        answer_id, qids = batch_dict["answer_id"].to(device), batch_dict["qid"]
        if dataset_name == "ivqa":
            answer_id = (answer_id / 2).clamp(max=1)
            answer_id_expanded = answer_id.to(device)
        else:
            answer_id_expanded = answer_id.view(-1, 1).expand_as(G_pred).to(device)

        G_map = mean_average_precision(G_pred, answer_id)
        D_map = mean_average_precision(D_pred, answer_id)

        agreeings = {"G": {}, "D": {}}
        for x in thresholds:
            if dataset_name == "ivqa":
                G_predicted = F.one_hot(G_pred[:, :x], num_classes=answer_id_expanded.size(1)).sum(1)
                D_predicted = F.one_hot(D_pred[:, :x], num_classes=answer_id_expanded.size(1)).sum(1)
                agreeings["G"][x] = (G_predicted * answer_id_expanded).max(1)[0]
                agreeings["D"][x] = (D_predicted * answer_id_expanded).max(1)[0]
            else:
                agreeings["G"][x] = G_pred[:, :x] == answer_id_expanded[:, :x]
                agreeings["D"][x] = D_pred[:, :x] == answer_id_expanded[:, :x]

        for i, qid in enumerate(qids):
            res[qid] = {"G_map": G_map.item(), "D_map": D_map.item()}
            for x in thresholds:
                res[qid][f"G_acc{x}"] = agreeings["G"][x][i].sum().detach().cpu().item()
                res[qid][f"D_acc{x}"] = agreeings["D"][x][i].sum().detach().cpu().item()

        dico = {"G_acc": agreeings["G"][1].sum() / len(qids), "D_acc": agreeings["D"][1].sum() / len(qids)}
        dico_reduced = dist.reduce_dict(dico)
        G_acc, D_acc = dico_reduced["G_acc"].item(), dico_reduced["D_acc"].item()
        metric_logger.update(G_acc=G_acc, D_acc=D_acc)

    all_res = dist.all_gather(res)
    results = reduce(lambda a, b: a.update(b) or a, all_res, {})
    assert len(results) == len(data_loader.dataset)
    out = {}
    for x in thresholds:
        out[f"G_acc{x}"] = sum(results[qid][f"G_acc{x}"] for qid in results) / len(results)
        out[f"D_acc{x}"] = sum(results[qid][f"D_acc{x}"] for qid in results) / len(results)
    out["G_map"] = sum(results[qid][f"G_map"] for qid in results) / len(results)
    out["D_map"] = sum(results[qid][f"D_map"] for qid in results) / len(results)

    if dist.is_main_process():
        print(dataset_name)
        for tag in ["G", "D"]:
            for x in thresholds:
                print(f"{tag}_acc{x}: {out[f'{tag}_acc{x}']: .2%}")
            print(f"{tag}_map@10: {out[f'{tag}_map']: .2%}")
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
    tokenizer = get_tokenizer()
    
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
    G, D = G_model(args), D_model(args)
    G.to(device)
    D.to(device)

    # Set up optimizer
    params_for_optimization = list(p for p in G.parameters() if p.requires_grad) + list(p for p in D.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(params_for_optimization, lr=args.lr)

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        G.load_state_dict(checkpoint["model"], strict=False)
        D.load_state_dict(checkpoint["model"], strict=False)

    a2tok = torch.zeros(len(dataloader_test.dataset.a2id), args.max_atokens).long()
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
        a2tok[aid] = tok
    a2tok = a2tok.to(G.device)
    G.set_answer_embeddings(a2tok, freeze_last=args.freeze_last)     # init answer embedding module
    if not args.eval:
        if dist.is_main_process():
            print("Start training")
        start_time = time.time()
        best_epoch = 0
        best_acc = 0
        for epoch in range(args.epochs):
            if dist.is_main_process():
                print(f"Starting epoch {epoch}")
            if args.distributed:
                sampler_train.set_epoch(epoch)
            train_stats = train_one_epoch(
                G=G,
                D=D,
                a2tok=a2tok,
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
                    G=G,
                    D=D,
                    a2tok=a2tok,
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
                if out["D_acc1"] > best_acc:
                    best_epoch = epoch
                    best_acc = out["D_acc1"]

                    if dist.is_main_process() and args.save_dir:
                        checkpoint_path = os.path.join(
                            args.save_dir, f"best_model.pth"
                        )
                        dist.save_on_master(
                            {
                                "G": G.state_dict(),
                                "D": D.state_dict()
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
                "epoch": epoch
            }

            if args.save_dir and dist.is_main_process():
                with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                checkpoint_path = os.path.join(args.save_dir, f"ckpt.pth")
                dist.save_on_master(
                    {
                        "G": G.state_dict(),
                        "D": D.state_dict()
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
            G.load_state_dict(checkpoint["G"], strict=False)
            D.load_state_dict(checkpoint["D"], strict=False)

    results, out = evaluate(
        G=G,
        D=D,
        a2tok=a2tok,
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
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    main(args)
