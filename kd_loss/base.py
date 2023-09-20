import torch
import torch.nn as nn
import torch.nn.functional as F
from .listwise import ListNet, STListNet, ListMLE, LambdaLoss
from .utils import pair_minus, Sinkhorn


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
        ind = torch.randperm(samples.size(1))[:self.num]
        return samples[:, ind]

    def exp_sample(self, samples):
        probs = self.rate * torch.exp(-self.rate * torch.arange(0, samples.size(1)))
        ind = torch.multinomial(probs, self.num)
        return samples[:, ind]

    def zipfs_sample(self, samples):
        Z = 1 / torch.arange(1, samples.size(1)+1)
        Z = Z / Z.sum()
        ind = torch.multinomial(Z, self.num)
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

    def sort_scores_by_teacher(self, t_score):
        sorted_ind = torch.argsort(t_score, descending=True)
        pos_list = sorted_ind[:, :self.n_pos]
        neg_list = self.Sampler(sorted_ind[:, self.n_pos:])
        select = torch.cat([pos_list, neg_list], 1)
        return select

    def forward(self, *args, **kwargs):
        pass

    def _loss_func(self, args):
        pass


class ListwiseLoss(RankingLoss):

    def __init__(self, args):
        super().__init__(args)

    def forward(self, t_score, s_score):
        """
        :param t_score: teacher predictions
        :param s_score: student predictions
        :return: kd_loss value, a torch.Tensor
        """
        select = self.sort_scores_by_teacher(t_score)
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
        self.loss_func = args.pair_loss

    def forward(self, gt, t_score, s_score, sample_dist, scalar=1000):
        bsz, n_ans = t_score.size()[:2]
        device = t_score.device
        uc_pair = pair_minus(sample_dist).std(1)
        mask = torch.eye(n_ans, device=device, dtype=torch.bool).expand(bsz, -1, -1)
        diag_values = torch.full((bsz*n_ans,), 1e10, device=device)
        uc_pair = uc_pair.masked_scatter(mask, diag_values)
        weight = Sinkhorn(uc_pair)
        weight = weight * scalar

        # take upper triangle
        mask = torch.ones_like(weight[0]).triu(1).bool()
        t_dist = pair_minus(t_score).masked_select(mask)
        s_dist = pair_minus(s_score).masked_select(mask)
        weight = weight.masked_select(mask)
        loss = self.loss(t_dist, s_dist, weight)
        return loss

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

    def _loss_func(self, args):
        loss_func = args.pair_loss
        if loss_func == 'margin_rank':
            return self._forward_margin_rank
        elif loss_func == 'ranknet':
            assert args.p_factor > 0
            return self._forward_ranknet
        else:
            raise NotImplementedError

