import torch
import torch.nn as nn
import torch.nn.functional as F
from .listwise import ListNet, STListNet, ListMLE, LambdaLoss
from .utils import pair_minus
from model.assigner import WeightAssigner


class Sampler(nn.Module):
    """
    Using different actors to sample negative labels.
    random: sample randomly
    exp: sample with an exponential distribution
    prob: sample with a given probability distribution
    """

    def __init__(self, num, actor, rate=0.005):
        super().__init__()
        self.num = num
        self.actor = actor
        self.rate = rate  # rate in exponential distribution sample. neg_num/(total_num*10)

    def forward(self, samples):
        assert samples.shape[1] >= self.num
        if self.actor == 'random':
            return self.random_sample(samples)
        elif self.actor == 'exp':
            return self.exp_sample(samples)
        elif self.actor == 'zipfs':
            return self.exp_sample(samples)
        else:
            return samples

    def random_sample(self, samples):
        ind = torch.randperm(samples.size(1))[:self.num].sort()[0]
        return samples[:, ind]

    def exp_sample(self, samples):
        probs = self.rate * torch.exp(-self.rate * torch.arange(0, samples.size(1)))
        ind = torch.multinomial(probs, self.num).sort()[0]
        return samples[:, ind]

    def zipfs_sample(self, samples):
        Z = 1 / torch.arange(1, samples.size(1)+1)
        Z = Z / Z.sum()
        ind = torch.multinomial(Z, self.num).sort()[0]
        return samples.gather(1, ind)


class RankingLoss(nn.Module):
    """
    This is the base model to construct the target ranking list in order to compute different losses.
    """
    def __init__(self, args):
        super().__init__()
        self.n_pos = args.n_pos
        self.Sampler = Sampler(args.n_neg, actor=args.neg_sampler)
        self.loss = self._loss_func(args)

    def sort_scores_by_teacher(self, gt, t_score, s_score):
        if len(gt.shape) == 1:
            sorted_ind = torch.argsort(t_score, descending=True)
            pos_list = sorted_ind[:, :self.n_pos]
            neg_list = self.Sampler(sorted_ind[:, self.n_pos:])
        else:
            pass
            # 从dataloader处理，保证每个batch中的num_gt一致
            # for gt, score in zip(gt, t_score):
            #     gt = gt.nonzero().squeeze()
            #     score[gt] = float('-inf')
            #     sorted_ind = torch.sort(score, descending=True).indices
            #     pos_list = []
        select = torch.cat([pos_list, neg_list], 1)
        return select

    def forward(self, *args, **kwargs):
        pass

    def _loss_func(self, args):
        pass


class ListwiseLoss(RankingLoss):

    def __init__(self, args):
        super().__init__(args)

    def forward(self, gt, t_score, s_score):
        """
        :param gt: ground-truth labels
        :param t_score: teacher predictions
        :param s_score: student predictions
        :return: kd_loss value, a torch.Tensor
        """
        select = self.sort_scores_by_teacher(gt, t_score, s_score)
        t_score = t_score.gather(1, select)
        s_score = s_score.gather(1, select)
        return self.loss(t_score, s_score)

    def _loss_func(self, args):
        loss_func = args.list_loss
        if loss_func == 'listnet':
            return ListNet(args.temperature)
        elif loss_func == 'stlistnet':
            return STListNet(args.temperature)
        elif loss_func == 'listmle':
            return ListMLE()
        elif loss_func == 'lambda':
            return LambdaLoss(args.lambda_weight, args.sigma, args.tau, args.temperature)
        else:
            raise NotImplementedError


class PairwiseLoss(RankingLoss):
    def __init__(self, args):
        super().__init__(args)
        self.factor = args.p_factor
        self.assigner = WeightAssigner(num_layer=args.gnn_layer,
                                       num_edge=args.num_edge,
                                       ds_node=args.ds_node,
                                       ds_edge=args.ds_edge,
                                       hidden_size=args.gnn_dim,
                                       dropout=args.gnn_dropout)

    def forward(self, gt, t_score, s_score, sample_dist, weight_decay=0):
        sample_rank = sample_dist.argsort(descending=True).argsort().float()
        uc_pair = pair_minus(sample_rank).std(1)
        uc_point = sample_rank.std(1)
        weight, select = self.assigner(sample_dist, uc_pair, uc_point)
        t_score = t_score.gather(1, select)
        s_score = s_score.gather(1, select)
        reg_loss = self.regularization(weight)

        # take upper triangle
        mask = torch.ones_like(weight[0]).triu(1).bool()
        t_dist = pair_minus(t_score).masked_select(mask)
        s_dist = pair_minus(s_score).masked_select(mask)
        weight = weight.masked_select(mask)
        loss = self.loss(t_dist, s_dist, weight)
        return loss + weight_decay * reg_loss

    def _forward_margin_rank(self, target, score, weight):
        target = target.sign()
        loss = torch.max(torch.zeros_like(score), -target*score + self.factor * weight).mean()
        return loss

    def _forward_ranknet(self, target, score, weight):
        target = (target.sign() + 1) / 2
        score = score / self.factor
        loss = F.binary_cross_entropy_with_logits(score, target, reduction='none')
        loss = (loss * weight).mean()
        return loss

    def regularization(self, weight):
        return 1 / weight.norm(dim=(1, 2)).mean()

    def _loss_func(self, args):
        loss_func = args.pair_loss
        if loss_func == 'margin_rank':
            return self._forward_margin_rank
        elif loss_func == 'ranknet':
            assert args.p_factor > 0
            return self._forward_ranknet
        else:
            raise NotImplementedError


# class PairwiseLoss(RankingLoss):
#     def __init__(self, args):
#         super().__init__(args)
#         self.factor = args.p_factor
#
#     def forward(self, gt, t_score, s_score, sample_dist):
#         select = self.sort_scores_by_teacher(gt, t_score, s_score)
#         t_score = t_score.gather(1, select)
#         s_score = s_score.gather(1, select)
#         sample_dist = sample_dist.gather(2, select.unsqueeze(1).expand(-1, sample_dist.size(1), -1))
#
#         uc = (pair_minus(sample_dist) > 0).float().mean(1)
#         select_ind, select_uc = self.sample(uc)
#         t_dist = pair_minus(t_score)[select_ind[:, 0], select_ind[:, 1], select_ind[:, 2]]
#         s_dist = pair_minus(s_score)[select_ind[:, 0], select_ind[:, 1], select_ind[:, 2]]
#         return self.loss_func(t_dist, s_dist, select_uc) * 2     # scale == 2
#
#     def _forward_margin_rank(self, target, score, weight):
#         target = target.sign()
#         loss = torch.max(torch.zeros_like(score), -target*score + self.factor * weight).mean()
#         return loss
#
#     def _forward_margin_mse(self, target, score, weight):
#         target = target * weight
#         score = score * weight
#         loss = F.mse_loss(score, target)
#         return loss
#
#     def _forward_ranknet(self, target, score, weight):
#         target = (target.sign() + 1) / 2
#         score = score / self.factor
#         loss = F.binary_cross_entropy_with_logits(score, target, weight=weight)
#         return loss
#
#     def _loss_func(self, args):
#         loss_func = args.pair_loss
#         if loss_func == 'margin_rank':
#             return self._forward_margin_rank
#         elif loss_func == 'margin_mse':
#             return self._forward_margin_mse
#         elif loss_func == 'ranknet':
#             assert args.p_factor > 0
#             return self._forward_ranknet
#         else:
#             raise NotImplementedError
