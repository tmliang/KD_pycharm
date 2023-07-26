import torch
import torch.nn as nn
import torch.nn.functional as F
from .listwise import ListNet, STListNet, ListMLE, LambdaLoss
from .utils import pair_minus


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


class ListwiseLoss(nn.Module):
    """
    This is the base model to construct the target ranking list in order to compute different listwise losses.
    """

    def __init__(self, args):
        super().__init__()
        self.n_pos = args.n_pos
        self.Sampler = Sampler(args.n_neg, actor=args.neg_sampler)
        self.loss = self._loss_func(args)

    def sort_scores_by_teacher(self, gt, t_score, s_score):
        if len(gt.shape) == 1:
            sorted_ind = torch.sort(t_score, descending=True).indices
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
        ind = torch.cat([pos_list, neg_list], 1)
        t_score = t_score.gather(1, ind)
        s_score = s_score.gather(1, ind)
        return t_score, s_score

    def forward(self, gt, t_score, s_score):
        """
        :param gt: ground-truth labels
        :param t_score: teacher predictions
        :param s_score: student predictions
        :return: kd_loss value, a torch.Tensor
        """
        t_score, s_score = self.sort_scores_by_teacher(gt, t_score, s_score)
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


class PairwiseLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_func = args.pair_loss
        self.factor = args.p_factor
        if args.pair_loss == 'margin_rank':
            self.loss_func = self._forward_margin_rank
        elif args.pair_loss == 'margin_mse':
            self.loss_func = self._forward_margin_mse
        elif args.pair_loss == 'ranknet':
            assert args.p_factor > 0
            self.loss_func = self._forward_ranknet

    def forward(self, t_score, s_score, sample_distances, hard=True):
        return self.loss_func(t_score, s_score, sample_distances, hard) * 2     # scale == 2

    def _forward_margin_rank(self, t_score, s_score, sample_distances, hard):
        if hard:
            sample_pos_pairs = (pair_minus(sample_distances) > 0).all(dim=1)
            t_pos_pairs = pair_minus(t_score) > 0
            select = (sample_pos_pairs & t_pos_pairs).float().nonzero()
            si = s_score[select[:, 0], select[:, 1]]
            sj = s_score[select[:, 0], select[:, 2]]
            loss = F.margin_ranking_loss(si, sj, torch.ones_like(si), margin=self.factor)
        else:
            sample_pos_prob = (pair_minus(sample_distances) > 0).float().mean(1)
            sample_pos_prob_norm = (sample_pos_prob - sample_pos_prob.transpose(1, 2)).triu_()
            target = pair_minus(t_score).sign().triu_()
            score = pair_minus(s_score).triu_()
            loss = torch.max(torch.zeros_like(score), -target*score + self.factor * sample_pos_prob_norm).mean()
        return loss

    def _forward_margin_mse(self, t_score, s_score, sample_distances, hard):
        t_dist = pair_minus(t_score)
        s_dist = pair_minus(s_score)
        if hard:
            sample_pos_pairs = (pair_minus(sample_distances) > 0).all(dim=1)
            select = sample_pos_pairs & (t_dist > 0)
            loss = F.mse_loss(s_dist[select], t_dist[select])
        else:
            sample_pos_prob_norm = (pair_minus(sample_distances) > 0).float().mean(1).triu_(1)    # one-side prob
            target = t_dist * sample_pos_prob_norm
            score = s_dist * sample_pos_prob_norm
            loss = F.mse_loss(score, target)
        return loss

    def _forward_ranknet(self, t_score, s_score, sample_distances, hard):
        t_dist = pair_minus(t_score)
        s_dist = pair_minus(s_score)
        if hard:
            sample_pos_pairs = (pair_minus(sample_distances) > 0).all(dim=1)
            select = sample_pos_pairs & (t_dist > 0)
            score = torch.sigmoid(s_dist[select] / self.factor)
            loss = F.binary_cross_entropy_with_logits(score, torch.ones_like(score))
        else:
            sample_pos_prob_norm = (pair_minus(sample_distances) > 0).float().mean(1).triu_(1)    # one-side prob
            target = (t_dist.sign() + 1) / 2
            score = s_dist / self.factor
            loss = F.binary_cross_entropy_with_logits(score, target, weight=sample_pos_prob_norm)
        return loss
