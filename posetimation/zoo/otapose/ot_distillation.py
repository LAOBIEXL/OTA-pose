import torch
import torch.nn.functional as F
import torch.nn as nn

class OTDistillationLoss(nn.Module):
    """Masked optimal-transport distillation with feature and position costs."""
    def __init__(self, lambda_pos=0.1, blur=0.05, max_iter=100):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.blur = blur
        self.max_iter = max_iter

    def compute_feature_cost(self, teacher_tokens, student_tokens):
        """
        teacher_tokens: [B, N, d]
        student_tokens: [B, M, d]
        """
        # 归一化
        teacher_norm = F.normalize(teacher_tokens, p=2, dim=-1)
        student_norm = F.normalize(student_tokens, p=2, dim=-1)

        # 计算余弦相似度矩阵
        cosine_sim = torch.bmm(teacher_norm, student_norm.transpose(1, 2))

        # 转换为距离（1 - 相似度）
        feature_cost = 1 - cosine_sim

        return feature_cost

    def compute_position_cost(self, teacher_pos, student_pos):
        """
        teacher_pos: [B, N, 3]  # (h, w, t)
        student_pos: [B, M, 3]
        """
        # 扩展维度
        teacher_pos_expanded = teacher_pos.unsqueeze(2)
        student_pos_expanded = student_pos.unsqueeze(1)

        # 欧式距离
        pos_diff = teacher_pos_expanded - student_pos_expanded
        position_cost = torch.sum(pos_diff ** 2, dim=-1)

        return position_cost

    def _valid_pair_max(self, values, pair_mask):
        masked_values = values.masked_fill(~pair_mask, float("-inf"))
        max_values = masked_values.amax(dim=(1, 2), keepdim=True)
        return torch.where(torch.isinf(max_values), torch.zeros_like(max_values), max_values)

    def compute_total_cost_matrix(self, teacher_tokens, student_tokens, teacher_pos, student_pos, pair_mask=None):
        """
        计算总代价矩阵
        """
        # 特征代价
        C_feat = self.compute_feature_cost(teacher_tokens, student_tokens)

        # 位置代价
        C_pos = self.compute_position_cost(teacher_pos, student_pos)

        if pair_mask is None:
            pair_mask = torch.ones_like(C_pos, dtype=torch.bool)

        # Per-sample, valid-pair normalization keeps each sample independent
        # from other batch elements and from padded teacher/student tokens.
        C_pos_max = self._valid_pair_max(C_pos, pair_mask)
        C_feat_max = self._valid_pair_max(C_feat, pair_mask)
        C_pos_normalized = C_pos / (C_pos_max + 1e-8) * C_feat_max

        # 总代价
        C_total = C_feat + self.lambda_pos * C_pos_normalized

        return C_total

    def _pair_mask(self, teacher_mask, student_mask, shape, device):
        B, N, M = shape
        if teacher_mask is None:
            teacher_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        else:
            teacher_mask = teacher_mask.to(device=device, dtype=torch.bool)
        if student_mask is None:
            student_mask = torch.ones(B, M, device=device, dtype=torch.bool)
        else:
            student_mask = student_mask.to(device=device, dtype=torch.bool)
        return teacher_mask, student_mask, teacher_mask[:, :, None] & student_mask[:, None, :]

    def sinkhorn_knopp(self, cost_matrix, teacher_mask=None, student_mask=None):
        B, N, M = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype
        teacher_mask, student_mask, pair_mask = self._pair_mask(teacher_mask, student_mask, (B, N, M), device)

        # 边际约束
        teacher_weight = teacher_mask.to(dtype)
        student_weight = student_mask.to(dtype)
        a = teacher_weight / teacher_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        b = student_weight / student_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)

        u = torch.ones_like(a)
        v = torch.ones_like(b)
        K = torch.exp(-cost_matrix / max(self.blur, 1e-8)) * pair_mask.to(dtype)

        for _ in range(self.max_iter):
            u = a / (torch.bmm(K, v) + 1e-8)
            v = b / (torch.bmm(K.transpose(1, 2), u) + 1e-8)

        P = u * K * v.transpose(1, 2)
        return P

    def forward(self, teacher_tokens, student_tokens, teacher_pos=None, student_pos=None, teacher_mask=None, student_mask=None):
        """
        计算OT损失
        返回: (ot_loss, feat_max)
        """
        B, N, d = teacher_tokens.shape
        B, M, d = student_tokens.shape
        teacher_mask, student_mask, pair_mask = self._pair_mask(teacher_mask, student_mask, (B, N, M), teacher_tokens.device)

        # 计算代价矩阵
        if teacher_pos is not None and student_pos is not None:
            cost_matrix = self.compute_total_cost_matrix(
                teacher_tokens,
                student_tokens,
                teacher_pos,
                student_pos,
                pair_mask=pair_mask,
            )
        else:
            cost_matrix = self.compute_feature_cost(teacher_tokens, student_tokens)
        cost_matrix = cost_matrix.masked_fill(~pair_mask, 0.0)

        # 记录feat_max（cost_matrix的最大值）
        valid_cost = cost_matrix.masked_fill(~pair_mask, float("-inf"))
        feat_max = valid_cost.amax().detach()
        if torch.isinf(feat_max):
            feat_max = cost_matrix.new_zeros(()).detach()

        # 求解最优传输计划
        P = self.sinkhorn_knopp(cost_matrix, teacher_mask=teacher_mask, student_mask=student_mask)

        # 按样本求和再取均值，避免loss尺度随batch/token数剧烈变化
        ot_loss = (P * cost_matrix).sum(dim=(1, 2)).mean()

        return ot_loss, feat_max

    def compute_position(self, tokens, token_shape, frame_idx=0):
        """
        从token恢复位置信息
        """
        B, N, d = tokens.shape
        H, W = token_shape

        # 假设tokens是按行优先排列的
        h_pos = torch.arange(H, device=tokens.device).repeat(W)
        w_pos = torch.arange(W, device=tokens.device).repeat_interleave(H)
        t_pos = torch.full((N,), frame_idx, device=tokens.device)

        pos = torch.stack([h_pos, w_pos, t_pos], dim=1)
        pos = pos.unsqueeze(0).repeat(B, 1, 1)

        return pos
