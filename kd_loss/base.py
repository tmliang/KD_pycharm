import random
import torch
import torch.nn as nn


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


class BaseLoss(nn.Module):
    """
    This is the base model to construct the target ranking list in order to compute kd_loss with different methods.
    """

    def __init__(self, n_pos=10, n_neg=50, neg_sampler='random'):
        super().__init__()
        self.n_pos = n_pos
        self.Sampler = Sampler(n_neg, actor=neg_sampler)

    def sort_scores_by_teacher(self, gt, t_score, s_score):
        if len(gt.shape) == 1:
            sorted_ind = torch.sort(t_score, descending=True).indices
            pos_list = sorted_ind[:, :self.n_pos]
            # random positive sample

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
        pass
