import argparse
import os

PRESAVE_DIR = "ckpt"
DATA_DIR = "data"
name2folder = {
    "webvid": "WebVid",
    "lsmdc": "LSMDC",
    "ivqa": "iVQA",
    "msrvtt": "MSRVTT-QA",
    "msvd": "MSVD-QA",
    "activitynet": "ActivityNet-QA",
    "tgif": "TGIF-QA",
    "how2qa": "How2QA",
    "tvqa": "TVQA"
}


def get_args_parser():
    parser = argparse.ArgumentParser("Set FrozenBiLM", add_help=False)

    # Dataset specific
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--ivqa_features_path",
        default=os.path.join(DATA_DIR, name2folder["ivqa"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--ivqa_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["ivqa"], "train.csv"),
    )
    parser.add_argument(
        "--ivqa_val_csv_path",
        default=os.path.join(DATA_DIR, name2folder["ivqa"], "val.csv"),
    )
    parser.add_argument(
        "--ivqa_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["ivqa"], "test.csv"),
    )
    parser.add_argument(
        "--ivqa_vocab_path",
        default=os.path.join(DATA_DIR, name2folder["ivqa"], "vocab.json"),
    )
    parser.add_argument(
        "--ivqa_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["ivqa"], "subtitles.pkl"),
    )
    parser.add_argument(
        "--msrvtt_features_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--msrvtt_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "train.csv"),
    )
    parser.add_argument(
        "--msrvtt_val_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "val.csv"),
    )
    parser.add_argument(
        "--msrvtt_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "test.csv"),
    )
    parser.add_argument(
        "--msrvtt_vocab_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "vocab.json"),
    )
    parser.add_argument(
        "--msrvtt_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["msrvtt"], "subtitles.pkl"),
    )
    parser.add_argument(
        "--msvd_features_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--msvd_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "train.csv"),
    )
    parser.add_argument(
        "--msvd_val_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "val.csv"),
    )
    parser.add_argument(
        "--msvd_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "test.csv"),
    )
    parser.add_argument(
        "--msvd_vocab_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "vocab.json"),
    )
    parser.add_argument(
        "--msvd_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["msvd"], "subtitles.pkl"),
    )
    parser.add_argument(
        "--activitynet_features_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--activitynet_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "train.csv"),
    )
    parser.add_argument(
        "--activitynet_val_csv_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "val.csv"),
    )
    parser.add_argument(
        "--activitynet_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "test.csv"),
    )
    parser.add_argument(
        "--activitynet_vocab_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "vocab.json"),
    )
    parser.add_argument(
        "--activitynet_subtitles_path",
        default=os.path.join(DATA_DIR, name2folder["activitynet"], "subtitles.pkl"),
    )
    parser.add_argument(
        "--tgif_features_path",
        default=os.path.join(DATA_DIR, name2folder["tgif"], "clipvitl14.pth"),
    )
    parser.add_argument(
        "--tgif_frameqa_train_csv_path",
        default=os.path.join(DATA_DIR, name2folder["tgif"], "train_frameqa.csv"),
    )
    parser.add_argument(
        "--tgif_frameqa_test_csv_path",
        default=os.path.join(DATA_DIR, name2folder["tgif"], "test_frameqa.csv"),
    )
    parser.add_argument(
        "--tgif_vocab_path",
        default=os.path.join(DATA_DIR, name2folder["tgif"], "vocab.json"),
    )

    # Training hyper-parameters
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="Adam optimizer parameter"
    )
    parser.add_argument(
        "--beta2", default=0.95, type=float, help="Adam optimizer parameter"
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch size used for training"
    )
    parser.add_argument(
        "--batch_size_val",
        default=32,
        type=int,
        help="batch size used for eval",
    )
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of training epochs"
    )
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        choices=["", "linear_with_warmup", "cosine_annealing_with_warmup"],
        help="learning rate decay schedule, default is constant",
    )
    parser.add_argument(
        "--fraction_warmup_steps",
        default=0.1,
        type=float,
        help="fraction of number of steps used for warmup when using linear schedule",
    )
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" epochs',
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10000,
        help="print log every print_freq iterations",
    )

    # Model parameters
    parser.add_argument(
        "--ds_factor_attn",
        type=int,
        default=8,
        help="downsampling factor for adapter attn",
    )
    parser.add_argument(
        "--ds_factor_ff",
        type=int,
        default=8,
        help="downsampling factor for adapter ff",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="dropout to use in the adapter"
    )
    parser.add_argument(
        "--ft_last",
        dest="freeze_last",
        action="store_false",
        help="whether to finetune answer embedding module or not",
    )

    # Run specific
    parser.add_argument(
        "--eval",
        action="store_true"
    )
    parser.add_argument(
        "--save_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--presave_dir",
        default=PRESAVE_DIR,
        help="the actual save_dir is an union of presave_dir and save_dir",
    )
    parser.add_argument("--device", default="cuda", help="device to use")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--load",
        default="",
        help="path to load checkpoint",
    )
    parser.add_argument(
        "--teacher_load",
        default="",
        help="path to load checkpoint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue training if loading checkpoint",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--num_workers", default=8, type=int, help="number of workers for dataloader"
    )

    # Distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    # Video and Text parameters
    parser.add_argument(
        "--max_feats",
        type=int,
        default=10,
        help="maximum number of video features considered, one per frame",
    )
    parser.add_argument(
        "--features_dim",
        type=int,
        default=768,
        help="dimension of the visual embedding space",
    )
    parser.add_argument(
        "--no_video",
        dest="use_video",
        action="store_false",
        help="disables usage of video",
    )
    parser.add_argument(
        "--no_context",
        dest="use_context",
        action="store_false",
        help="disables usage of speech",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="maximum number of tokens in the input text prompt",
    )
    parser.add_argument(
        "--max_atokens",
        type=int,
        default=5,
        help="maximum number of tokens in the answer",
    )
    parser.add_argument(
        "--prefix",
        default="",
        type=str,
        help="task induction before question for videoqa",
    )
    parser.add_argument(
        "--suffix",
        default=".",
        type=str,
        help="suffix after the answer mask for videoqa",
    )
    # Ranking parameters
    parser.add_argument(
        "--n_pos",
        type=int,
        default=10,
        help="maximum number of positive labels",
    )
    parser.add_argument(
        "--n_neg",
        type=int,
        default=100,
        help="maximum number of negative labels",
    )
    parser.add_argument(
        "--neg_sampler",
        type=str,
        default=None,
        choices=["random", "exp", "zipfs"],
        help="negatively sampling methods, None indicates use all negative labels"
    )
    parser.add_argument(
        "--lambda_weight",
        type=str,
        default='ndcg',
        choices=["ndcg", "ndcg1", "ndcg2", "ndcg2++"],
        help="weight scheme in lambda losses"
    )
    parser.add_argument(
        "--loss_func",
        type=str,
        default='listnet',
        choices=["vanilla",
                 "dkd", "stdkd", "tridkd",
                 "listnet", "stlistnet", "listmle", "lambda", "margin_mse", "margin_rank", "ranknet"
                 ]
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.,
        help="margin in margin-mse and margin-rank losses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.
    )
    parser.add_argument(
        "--uc_mode",
        type=str,
        default="",
        help="'l': learnable dropout, 'b': BNN, 'v': VAE."
    )
    parser.add_argument("--uc_drop", type=float, default=0)
    parser.add_argument("--num_sample", type=int, default=10)
    parser.add_argument("--alpha_r", default=10, type=float)
    parser.add_argument("--alpha_f", default=0, type=float)
    parser.add_argument("--alpha_uc", default=0, type=float, help="weight of standard bias")
    parser.add_argument("--uc_eval", action="store_true")

    parser.add_argument("--dropout_schedule",
        default="linear_with_warmup",
        choices=["", "linear_with_warmup", "cosine_annealing_with_warmup", "random"]
    )
    parser.add_argument("--min_drop", type=float, default=0)
    parser.add_argument("--max_drop", type=float, default=0.5)
    parser.add_argument("--drop_warmup_fraction", type=float, default=0.5)
    return parser
