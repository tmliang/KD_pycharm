from .listwise import ListNet, STListNet, ListMLE, LambdaLoss, LambdaRank
from .pairwise import MarginRank, MarginMSE, RankNet
from .logit_kd import Logit_KD
from .dkd import Decoupled_KD
from .st_dkd import STDecoupled_KD
from .tri_dkd import TriDecoupled_KD


def build_ranking_loss(args):
    n_pos = args.n_pos
    n_neg = args.n_neg
    neg_sampler = args.neg_sampler
    loss_func = args.loss
    if loss_func == 'listnet':
        return ListNet(n_pos, n_neg, neg_sampler, args.temperature)
    elif loss_func == 'stlistnet':
        return STListNet(n_pos, n_neg, neg_sampler, args.temperature)
    elif loss_func == 'listmle':
        return ListMLE(n_pos, n_neg, neg_sampler)
    elif loss_func == 'lambda':
        return LambdaLoss(n_pos, n_neg, neg_sampler,  args.lambda_weight, args.sigma, args.mu)
    elif loss_func == 'lambdarank':
        return LambdaRank(n_pos, n_neg, neg_sampler, args.sigma)
    elif loss_func == 'margin_mse':
        return MarginMSE(n_pos, n_neg, neg_sampler, args.margin)
    elif loss_func == 'margin_rank':
        return MarginRank(n_pos, n_neg, neg_sampler, args.margin)
    elif loss_func == 'ranknet':
        return RankNet(n_pos, n_neg, neg_sampler, args.sigma)
    elif loss_func == 'logit_kd':
        return Logit_KD(args.temperature)
    elif loss_func == 'dkd':
        return Decoupled_KD(args.temperature, args.dkd_alpha, args.dkd_beta)
    elif loss_func == 'stdkd':
        return STDecoupled_KD(n_pos, n_neg, neg_sampler, args.temperature, args.dkd_alpha, args.dkd_beta)
    elif loss_func == 'tridkd':
        return TriDecoupled_KD(n_pos, n_neg, neg_sampler, args.temperature, args.dkd_alpha, args.dkd_beta)
    else:
        raise NotImplementedError
