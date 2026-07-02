#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/6 0:02
# @Author : WeiHua

import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn
import torch.nn.functional as F
import os

# code inherited from "https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/ot_loss.py"
class OT_Loss(Module):
    def __init__(self, num_of_iter_in_ot=100, reg=10.0, method='sinkhorn'):
        super(OT_Loss, self).__init__()
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.method = method

    def forward(self, t_scores, s_scores, pts, cost_type='all', clamp_ot=True, aux_cost=None):
        """
        Calculating OT loss between teacher and student's distribution.
        Cost map is defined as: cost = dist(p_t, p_s) + dist(score_t, score_s).
        All dist are l2 distance.
        Args:
            t_scores: Tensor with shape (N, )
            s_scores: Tensor with shape (N, )

        Returns:

        """
        assert cost_type in ['all', 'dist', 'score']
        with torch.no_grad():
            t_scores_prob = torch.softmax(t_scores, dim=0)
            s_scores_prob = torch.softmax(s_scores, dim=0)
            score_cost = (t_scores.detach().unsqueeze(1) - s_scores.detach().unsqueeze(0)) ** 2
            score_cost = score_cost / score_cost.max()

            if cost_type in ['all', 'dist']:
                coord_x = pts[:, 0]
                coord_y = pts[:, 1]
                dist_x = (coord_x.reshape(1, -1) - coord_x.reshape(-1, 1)) ** 2
                dist_y = (coord_y.reshape(1, -1) - coord_y.reshape(-1, 1)) ** 2
                dist_cost = (dist_x + dist_y).to(t_scores_prob.device)
                dist_cost = dist_cost / dist_cost.max()
                if cost_type == 'all':
                    cost_map = dist_cost + score_cost
                else:
                    cost_map = dist_cost
            else:
                cost_map = score_cost

            if not isinstance(aux_cost, type(None)):
                cost_map = cost_map + aux_cost

            # cost_map = cost_map * 100

            source_prob = s_scores_prob.detach().view(-1)
            target_prob = t_scores_prob.detach().view(-1)


            _, log = sinkhorn(target_prob.cpu(), source_prob.cpu(), cost_map.cpu(), self.reg,
                                maxIter=self.num_of_iter_in_ot, log=True,
                                method=self.method)

            beta = log['beta'].to(target_prob.device)  # size is the same as source_prob: [#cood * #cood]
            alpha = log['alpha'].to(target_prob.device)

            #beta就是文章中的u


        # teacher_density = t_scores.detach()
        # teacher_count = teacher_density.sum()
        #
        # source_density = s_scores.detach()
        # source_count = source_density.sum()
        #
        # loss = torch.sum(alpha * (teacher_density / teacher_count)) + torch.sum(beta * source_count * (source_density / source_count))



        # compute the gradient of OT loss to predicted density (unnormed_density).
        # im_grad = beta / source_count - < beta, source_density> / (source_count)^2

        source_density = s_scores.detach().view(-1)
        source_count = source_density.sum()

        im_grad_1 = (source_count) / (source_count * source_count + 1e-8) * beta  # size of [#cood * #cood]

        im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8)  # size of 1

        im_grad = im_grad_1 - im_grad_2

        im_grad = im_grad.detach()

        # # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t predicted density is im_grad.

        loss = torch.sum(s_scores * im_grad) * 10
        non_negative_loss = torch.clamp_min(loss, 0)

        # print('-------------------------------')
        # print('OT loss: {}'.format(non_negative_loss))
        # print('-------------------------------')

        return non_negative_loss

class Cont_Loss(Module):
    def __init__(self, epsilon=0.05, sinkhorn_iterations=3,
                 method='distributed_sinkhorn', world_size=None, temperature=0.1,
                 w_cross=1.0, w_same=1.0, w_student=1.0, w_center=1.0,
                 queue_length=0, feat_num=3000):
        """
        Args:
            epsilon: Sinkhorn regularization
            sinkhorn_iterations: iterations of distributed_sinkhorn
            method: 'distributed_sinkhorn' etc.
            world_size: auto-detect if None
            temperature: softmax temperature
            w_cross: weight for cross-domain (A↔B)
            w_same: weight for same-domain teacher-student (A↔A, B↔B)
            w_student: weight for student-student alignment
        """
        super(Cont_Loss, self).__init__()
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.method = method
        self.temperature = temperature
        self.w_cross = w_cross
        self.w_same = w_same
        self.w_student = w_student
        self.w_center = w_center

        if world_size is None:
            self.world_size = 1
        else:
            self.world_size = world_size

        self.queue_length = queue_length
        self.feat_dim = feat_num

        if queue_length > 0:  
            self.register_buffer("queue_A", torch.zeros(queue_length, feat_num))
            self.register_buffer("queue_B", torch.zeros(queue_length, feat_num))
            self.register_buffer("queue_center_t", torch.zeros(queue_length, feat_num))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _sinkhorn_assign(self, x):
        with torch.no_grad():
            q = sinkhorn(
                a=None,
                b=None,
                C=x, 
                method='distributed_sinkhorn',
                world_size=self.world_size
            ).detach()
        return q

    def _softmax_prob(self, x):
        return F.softmax(x / self.temperature, dim=1)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, feats_A, feats_B, t_center):
        if self.queue_length == 0:
            return

        bs = feats_A.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + bs

        if end <= self.queue_length:
            self.queue_A[ptr:end] = feats_A
            self.queue_B[ptr:end] = feats_B
            self.queue_center_t[ptr:end] = t_center
        else:
            overflow = end - self.queue_length
            self.queue_A[ptr:] = feats_A[:bs - overflow]
            self.queue_B[ptr:] = feats_B[:bs - overflow]
            self.queue_center_t[ptr:] = t_center[:bs - overflow]
            self.queue_A[:overflow] = feats_A[bs - overflow:]
            self.queue_B[:overflow] = feats_B[bs - overflow:]
            self.queue_center_t[:overflow] = t_center[bs - overflow:]

        self.queue_ptr[0] = (ptr + bs) % self.queue_length


    def forward(self, t_pros, s_pros, t_center, s_center, use_queue):
        """
        Args:
            t_pros: Tensor [2N, K] — two domains of teacher 
            s_pros: Tensor [2N, K] — two domains of student 
            t_center : Tensor [N, K] — domain center of teacher
            s_center : Tensor [N, K] — domain center of student
        Returns:
            scalar loss 
        """
        N = t_pros.shape[0] // 2
        t_A, t_B = torch.split(t_pros, N, dim=0)
        s_A, s_B = torch.split(s_pros, N, dim=0)

        use_the_queue = False
        if self.queue_length > 0 and self.queue_ptr.item() > 0 and use_queue:
            use_the_queue = True

        if use_the_queue:
            qA_feats = self.queue_A.detach().to(t_A.device)
            qB_feats = self.queue_B.detach().to(t_B.device)
            q_center_t_feats = self.queue_center_t.detach().to(t_center.device)
            t_A_full = torch.cat([qA_feats, t_A], dim=0)
            t_B_full = torch.cat([qB_feats, t_B], dim=0)
            t_center_full = torch.cat([q_center_t_feats, t_center], dim=0)
        else:
            t_A_full, t_B_full, t_center_full = t_A, t_B, t_center
        # Teacher assignment
        q_A_full = self._sinkhorn_assign(t_A_full)
        q_B_full = self._sinkhorn_assign(t_B_full)
        q_center_full = self._sinkhorn_assign(t_center_full)
        q_A, q_B, q_center = q_A_full[-N:], q_B_full[-N:], q_center_full[-N:]

        # Student probability
        p_A = self._softmax_prob(s_A)
        p_B = self._softmax_prob(s_B)
        p_center = self._softmax_prob(s_center)

        # Cross-domain teacher student
        loss_cross_AB = -torch.mean(torch.sum(q_A * torch.log(p_B + 1e-8), dim=1))
        loss_cross_BA = -torch.mean(torch.sum(q_B * torch.log(p_A + 1e-8), dim=1))
        loss_cross = (loss_cross_AB + loss_cross_BA) / 2

        # Same-domain teacher student
        loss_same_A = -torch.mean(torch.sum(q_A * torch.log(p_A + 1e-8), dim=1))
        loss_same_B = -torch.mean(torch.sum(q_B * torch.log(p_B + 1e-8), dim=1))
        loss_same = (loss_same_A + loss_same_B) / 2

        # # Student↔Student
        # sim_student = torch.matmul(p_A, p_B.T)  # [N,N]
        # sim_student = sim_student / self.temperature
        # targets = torch.arange(N, device=p_A.device)
        # loss_student = (F.cross_entropy(sim_student, targets) +
        #                 F.cross_entropy(sim_student.T, targets)) / 2
        
        # center
        smooth = 0.05
        q_center_smooth = (1 - smooth) * q_center + smooth / q_center.shape[1]
        loss_agg = F.kl_div(torch.log(p_center + 1e-8), q_center_smooth.detach(), reduction='batchmean')

        # update queue
        self._dequeue_and_enqueue(t_A.detach(), t_B.detach(), t_center.detach())

        w = self.w_cross + self.w_same + self.w_center
        # total loss
        loss = (self.w_cross * loss_cross +
                self.w_same * loss_same +
                self.w_center * loss_agg) / w

        return loss
