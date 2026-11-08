import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricChamferDistillationLoss(nn.Module):
    """Hard bidirectional nearest-neighbor matching for token distillation.

    The pairwise cost mirrors the existing OT-v3 cost: cosine feature distance
    plus optional normalized squared position distance. Unlike OT, each token
    independently chooses its nearest neighbor, so the matching is many-to-one
    and has no mass constraint.
    """

    def __init__(self, lambda_pos=0.1):
        super().__init__()
        self.lambda_pos = float(lambda_pos)
        if not math.isfinite(self.lambda_pos) or self.lambda_pos < 0:
            raise ValueError("lambda_pos must be finite and non-negative")

    @staticmethod
    def _validate_mask(mask, batch_size, token_num, device, name):
        if mask is None:
            mask = torch.ones(batch_size, token_num, device=device, dtype=torch.bool)
        else:
            if mask.shape != (batch_size, token_num):
                raise ValueError(f"{name} must have shape [B, N]")
            mask = mask.to(device=device, dtype=torch.bool)
        if (~mask.any(dim=1)).any():
            raise ValueError(f"each sample must contain at least one valid {name.removesuffix('_mask')} token")
        return mask

    @staticmethod
    def _valid_pair_max(values, pair_mask):
        maximum = values.masked_fill(~pair_mask, -torch.inf).amax(dim=(1, 2), keepdim=True)
        return torch.where(torch.isinf(maximum), torch.zeros_like(maximum), maximum)

    def forward(
        self,
        teacher_tokens,
        student_tokens,
        teacher_pos=None,
        student_pos=None,
        teacher_mask=None,
        student_mask=None,
    ):
        if teacher_tokens.ndim != 3 or student_tokens.ndim != 3:
            raise ValueError("teacher_tokens and student_tokens must have shape [B, N, C]")
        batch_size, teacher_num, feature_dim = teacher_tokens.shape
        student_batch, student_num, student_dim = student_tokens.shape
        if batch_size != student_batch or feature_dim != student_dim:
            raise ValueError("teacher and student batch/feature dimensions must match")
        if teacher_num == 0 or student_num == 0:
            raise ValueError("teacher and student must each contain at least one token")

        teacher_mask = self._validate_mask(
            teacher_mask, batch_size, teacher_num, teacher_tokens.device, "teacher_mask"
        )
        student_mask = self._validate_mask(
            student_mask, batch_size, student_num, teacher_tokens.device, "student_mask"
        )
        pair_mask = teacher_mask[:, :, None] & student_mask[:, None, :]

        # Keep normalization and reductions in fp32 under AMP for stable min ops.
        with torch.autocast(device_type=teacher_tokens.device.type, enabled=False):
            teacher_input = teacher_tokens.float().masked_fill(~teacher_mask.unsqueeze(-1), 0)
            student_input = student_tokens.float().masked_fill(~student_mask.unsqueeze(-1), 0)
            if not torch.isfinite(teacher_input).all() or not torch.isfinite(student_input).all():
                raise ValueError("valid teacher and student token features must be finite")

            teacher = F.normalize(teacher_input, p=2, dim=-1)
            student = F.normalize(student_input, p=2, dim=-1)
            feature_cost = 1 - torch.bmm(teacher, student.transpose(1, 2))
            total_cost = feature_cost

            if (teacher_pos is None) != (student_pos is None):
                raise ValueError("teacher_pos and student_pos must be provided together")
            if teacher_pos is not None:
                if teacher_pos.shape[:2] != (batch_size, teacher_num):
                    raise ValueError("teacher_pos must have shape [B, N, P]")
                if student_pos.shape[:2] != (batch_size, student_num):
                    raise ValueError("student_pos must have shape [B, M, P]")
                if teacher_pos.shape[-1] != student_pos.shape[-1]:
                    raise ValueError("teacher and student position dimensions must match")

                teacher_position = teacher_pos.float().masked_fill(~teacher_mask.unsqueeze(-1), 0)
                student_position = student_pos.float().masked_fill(~student_mask.unsqueeze(-1), 0)
                if not torch.isfinite(teacher_position).all() or not torch.isfinite(student_position).all():
                    raise ValueError("valid teacher and student positions must be finite")

                position_cost = (
                    teacher_position.unsqueeze(2) - student_position.unsqueeze(1)
                ).square().sum(dim=-1)
                position_max = self._valid_pair_max(position_cost, pair_mask)
                feature_max = self._valid_pair_max(feature_cost, pair_mask)
                scaled_position = position_cost / (position_max + 1e-8) * feature_max
                total_cost = feature_cost + self.lambda_pos * scaled_position

            masked_cost = total_cost.masked_fill(~pair_mask, torch.inf)
            teacher_to_student = masked_cost.min(dim=2).values
            student_to_teacher = masked_cost.min(dim=1).values
            teacher_to_student = teacher_to_student.masked_fill(~teacher_mask, 0)
            student_to_teacher = student_to_teacher.masked_fill(~student_mask, 0)

            forward_loss = (
                (teacher_to_student * teacher_mask).sum(dim=1)
                / teacher_mask.sum(dim=1)
            )
            backward_loss = (
                (student_to_teacher * student_mask).sum(dim=1)
                / student_mask.sum(dim=1)
            )
            loss = (0.5 * (forward_loss + backward_loss)).mean()
            cost_max = self._valid_pair_max(total_cost, pair_mask).amax().detach()

        return loss, cost_max
