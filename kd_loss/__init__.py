from .base import RankingLoss
from .vanilla_kd import VanillaKD
from .relation import RKD, R2KD
from .decouple import Decoupled_KD
from .feature import FKD


def build_ranking_loss(args):
    if args.loss_func == "vanilla":
        return VanillaKD(args.temperature)
    elif "dkd" in args.loss_func:
        return Decoupled_KD()
    else:
        return RankingLoss(args)
