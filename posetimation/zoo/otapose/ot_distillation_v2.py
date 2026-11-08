import torch
import torch.nn.functional as F
import torch.nn as nn

class OTDistillationLoss(nn.Module):
    def __init__(self, lambda_pos=0.1, temperature=0.1):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.temperature = temperature
    
    def _valid_pair_max(self, values, pair_mask):
        masked_values = values.masked_fill(~pair_mask, float("-inf"))
        max_values = masked_values.amax(dim=(1, 2), keepdim=True)
        return torch.where(torch.isinf(max_values), torch.zeros_like(max_values), max_values)

    def compute_feature_cost(self, teacher_tokens, student_tokens, pair_mask=None):
        """
        teacher_tokens: [B, N, d]
        student_tokens: [B, M, d]
        """
        # 归一化
        teacher_norm = F.normalize(teacher_tokens, p=2, dim=-1)
        student_norm = F.normalize(student_tokens, p=2, dim=-1)
        
        # 计算余弦相似度矩阵
        cosine_sim = torch.bmm(teacher_norm, student_norm.transpose(1, 2))
        
        # 转换为距离（1 - 相似度）并归一化到[0, 1]
        feature_cost = 1 - cosine_sim
        if pair_mask is None:
            pair_mask = torch.ones_like(feature_cost, dtype=torch.bool)
        feature_cost = feature_cost / (self._valid_pair_max(feature_cost, pair_mask) + 1e-8)
        
        return feature_cost
    
    def compute_position_cost(self, teacher_pos, student_pos, pair_mask=None):
        """
        teacher_pos: [B, N, 3]  # (h, w, t)
        student_pos: [B, M, 3]
        """
        # 扩展维度
        teacher_pos_expanded = teacher_pos.unsqueeze(2)
        student_pos_expanded = student_pos.unsqueeze(1)
        
        # 使用L1距离替代L2距离
        position_cost = torch.sum(torch.abs(teacher_pos_expanded - student_pos_expanded), dim=-1)
        
        # 归一化位置代价
        if pair_mask is None:
            pair_mask = torch.ones_like(position_cost, dtype=torch.bool)
        position_cost = position_cost / (self._valid_pair_max(position_cost, pair_mask) + 1e-8)
        
        return position_cost
    
    def compute_total_cost_matrix(self, teacher_tokens, student_tokens, teacher_pos, student_pos, pair_mask=None):
        """
        计算总代价矩阵
        """
        # 特征代价
        C_feat = self.compute_feature_cost(teacher_tokens, student_tokens, pair_mask=pair_mask)
        
        # 位置代价
        C_pos = self.compute_position_cost(teacher_pos, student_pos, pair_mask=pair_mask)
        
        # 总代价
        C_total = C_feat + self.lambda_pos * C_pos
        
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

    def sinkhorn_knopp(self, cost_matrix, max_iter=50, epsilon=1e-3, teacher_mask=None, student_mask=None):
        """
        Sinkhorn-Knopp算法求解最优传输计划
        使用更稳定的迭代方式
        """
        B, N, M = cost_matrix.shape
        teacher_mask, student_mask, pair_mask = self._pair_mask(teacher_mask, student_mask, (B, N, M), cost_matrix.device)
        teacher_weight = teacher_mask.to(cost_matrix.dtype)
        student_weight = student_mask.to(cost_matrix.dtype)
        
        # 初始化行和列的缩放因子
        # u: [B, N, 1], v: [B, M, 1]
        u = torch.ones(B, N, 1, device=cost_matrix.device, dtype=cost_matrix.dtype)
        v = torch.ones(B, M, 1, device=cost_matrix.device, dtype=cost_matrix.dtype)
        
        # 计算核
        K = torch.exp(-cost_matrix / self.temperature) * pair_mask.to(cost_matrix.dtype)
        
        # 迭代更新
        for _ in range(max_iter):
            # 更新u: u = 1 / (K @ v)
            u = (1.0 / (torch.bmm(K, v) + 1e-8)) * teacher_weight[:, :, None]
            # 更新v: v = 1 / (K^T @ u)
            v = (1.0 / (torch.bmm(K.transpose(1, 2), u) + 1e-8)) * student_weight[:, :, None]
        
        # 计算传输计划
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
            cost_matrix = self.compute_feature_cost(teacher_tokens, student_tokens, pair_mask=pair_mask)
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
        
        # 生成网格位置
        h_pos, w_pos = torch.meshgrid(
            torch.arange(H, device=tokens.device),
            torch.arange(W, device=tokens.device),
            indexing='ij'
        )
        h_pos = h_pos.reshape(-1).float() / (H - 1) if H > 1 else h_pos.reshape(-1).float()
        w_pos = w_pos.reshape(-1).float() / (W - 1) if W > 1 else w_pos.reshape(-1).float()
        t_pos = torch.full((N,), frame_idx, device=tokens.device)
        
        pos = torch.stack([h_pos, w_pos, t_pos], dim=1)
        pos = pos.unsqueeze(0).repeat(B, 1, 1)
        
        return pos
