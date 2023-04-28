import os
from .deberta import DebertaV2ForMaskedLM, DebertaV2ForRTD
from .deberta_for_finetune import DebertaV2
from transformers import DebertaV2Tokenizer

TRANSFORMERS_CACHE = 'TRANSFORMERS_CACHE'
model_name = os.path.join(TRANSFORMERS_CACHE, 'deberta-v2-xlarge')


def joint_model(args):
    model = DebertaV2.from_pretrained(
        features_dim=args.features_dim if args.use_video else 0,
        max_feats=args.max_feats,
        ds_factor_attn=args.ds_factor_attn,
        ds_factor_ff=args.ds_factor_ff,
        dropout=args.dropout,
        n_ans=args.n_ans,
        freeze_last=args.freeze_last,
        pretrained_model_name_or_path=model_name,
        local_files_only=True,
    )
    return model


def G_model(args):
    model = DebertaV2ForMaskedLM.from_pretrained(
        features_dim=args.features_dim if args.use_video else 0,
        max_feats=args.max_feats,
        ds_factor_attn=args.ds_factor_attn,
        ds_factor_ff=args.ds_factor_ff,
        dropout=args.dropout,
        n_ans=args.n_ans,
        freeze_last=args.freeze_last,
        pretrained_model_name_or_path=model_name,
        local_files_only=True,
    )
    return model


def D_model(args):
    model = DebertaV2ForRTD.from_pretrained(
        features_dim=args.features_dim if args.use_video else 0,
        max_feats=args.max_feats,
        ds_factor_attn=args.ds_factor_attn,
        ds_factor_ff=args.ds_factor_ff,
        dropout=args.dropout,
        pretrained_model_name_or_path=model_name,
        local_files_only=True,
    )
    return model


def get_tokenizer():
    tokenizer = DebertaV2Tokenizer.from_pretrained(
        model_name, local_files_only=True
    )
    return tokenizer
