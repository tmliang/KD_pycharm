from .base import RankingLoss
from .vanilla_kd import VanillaKD
from .relation import RKD, R2KD
from .decouple import Decoupled_KD
from .feature import FKD
from .uncertainty.nce import NceLoss


def build_ranking_loss(args):
    loss0 = FKD(args)
    loss0.to(args.device)
    if args.loss1 == "vanilla":
        loss1 = VanillaKD(args.temperature)
    elif "dkd" in args.loss1:
        loss1 = Decoupled_KD()
    else:
        loss1 = RankingLoss(args)

    if args.loss2 == "rkd":
        loss2 = RKD()
    else:
        loss2 = R2KD(args)
    return loss0, loss1, loss2
