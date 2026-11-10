"""OTA-Pose with adaptive token pruning and cross-branch distillation."""
import logging
import math

import torch
import torch.nn as nn

from utils.utils_registry import MODEL_REGISTRY
from posetimation.layers import CHAIN_RSB_BLOCKS
from ..base import BaseModel
from ..backbones.vit import ViT
from ..layers.corss_block import CrossBlock
from .token_prune_pose import Block, Model, cluster_dpc_knn, compute_temporal_redundancy_stats
from .token_prune_dynamic import cluster_dpc_knn_dynamic
from .ot_distillation import OTDistillationLoss
from .chamfer_distillation import SymmetricChamferDistillationLoss

logger = logging.getLogger(__name__)

__all__ = ["OTAPose"]

class ModelArgs:
    st_layers: int = 2
    middle_layers: int = 2

    def __post_init__(self):
        pass

@MODEL_REGISTRY.register()
class OTAPose(BaseModel):

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = OTAPose(cfg, phase, **kwargs)
        return model

    def exists(self, x):
        return x is not None

    def default(self, val, d):
        if self.exists(val):
            return val
        return d() if callable(d) else d

    def __init__(self, cfg, phase, **kwargs):
        super().__init__()

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.heatmap_h = cfg.MODEL.HEATMAP_SIZE[1]
        self.heatmap_w = cfg.MODEL.HEATMAP_SIZE[0]
        self.freeze_backbone_weights = cfg.MODEL.FREEZE_BACKBONE_WEIGHTS
        self.token_num = int((cfg.MODEL.IMAGE_SIZE[1] / cfg.MODEL.VIT.PATCH_SIZE) * (
                cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.VIT.PATCH_SIZE))
        self.map_size_up_1x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE)
        self.map_size_up_2x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE * 2, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE * 2)
        self.map_size_up_4x = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.VIT.PATCH_SIZE * 4, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.VIT.PATCH_SIZE * 4)

        self.frame_num = cfg.MODEL.FRAME_NUM

        # self.multi_frame_fusion = True

        self.htms_dict = dict()
        # Last forward token-keep statistics for quick inspection/debug.
        self.token_keep_stats = dict()
        self.token_keep_history = {
            "fine_keep_tokens": [],
            "coarse_keep_tokens": [],
            "fine_keep_ratio": [],
            "coarse_keep_ratio": [],
        }

        self.backbone = ViT.get_net(cfg , phase)

        #细粒度剪枝：
        # 直接使用Transformer输出(1024维)，通过反卷积上采样4倍
        # 反卷积：将 (b×frames, 1024, H, W) → (b×frames, 1024, H×2, W×2)
        self.fine_grain_deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=cfg.MODEL.VIT.EMB_DIM,
                               out_channels=cfg.MODEL.VIT.EMB_DIM,
                               kernel_size=4, stride=2, padding=1,
                               output_padding=0, bias=False),
            nn.BatchNorm2d(cfg.MODEL.VIT.EMB_DIM),
            nn.ReLU(inplace=True),
        )
        dpr = [x.item() for x in torch.linspace(0, 0.06, 2)]
        self.fg_pruned_blocks = nn.ModuleList([
            Block(
                dim=cfg.MODEL.VIT.EMB_DIM, num_heads=8, mlp_hidden_dim=cfg.MODEL.VIT.EMB_DIM * 2,
                 drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(2)])
        self.fg_pruned_norm  = nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM)

        #粗粒度剪枝+恢复
        self.add_pos_emb = cfg.MODEL.EXTRA.ADD_POS_EMB
        self.pos_emb_list = nn.ParameterList()
        if self.add_pos_emb:
            for i in range(self.frame_num):
                pos_emb = nn.Parameter(torch.zeros(1, self.token_num, cfg.MODEL.VIT.EMB_DIM))
                nn.init.trunc_normal_(pos_emb, std=0.02)
                self.pos_emb_list.append(pos_emb)

        extra_cfg = getattr(cfg.MODEL, "EXTRA", None)
        # 是否拼接backbone特征到输出头（可通过配置控制）
        self.final_fusion_backbone = bool(getattr(extra_cfg, "FINAL_FUSION_BACKBONE", True))
        self.use_recover = True
        #当no recover时，多帧fusion其实就是无关的
        self.multi_frame_fusion = False
        self.coarse_prune_ratio = int(getattr(extra_cfg, "COARSE_PRUNE_RATIO", 10))
        self.fine_grain_ratio = int(getattr(extra_cfg, "FINE_GRAIN_RATIO", 10))
        self.coarse_prune_ratio = max(1, self.coarse_prune_ratio)
        self.fine_grain_ratio = max(1, self.fine_grain_ratio)
        self.static_coarse_keep_tokens = getattr(extra_cfg, "STATIC_COARSE_KEEP_TOKENS", None)
        self.static_fine_keep_tokens = getattr(extra_cfg, "STATIC_FINE_KEEP_TOKENS", None)
        if self.static_coarse_keep_tokens is not None:
            self.static_coarse_keep_tokens = max(1, int(self.static_coarse_keep_tokens))
        if self.static_fine_keep_tokens is not None:
            self.static_fine_keep_tokens = max(1, int(self.static_fine_keep_tokens))

        self.use_dynamic_prune = getattr(extra_cfg, "DYNAMIC_PRUNE", True)
        self.prune_selection = str(getattr(extra_cfg, "PRUNE_SELECTION", "score")).lower()
        if self.prune_selection not in ("score", "random"):
            raise ValueError("MODEL.EXTRA.PRUNE_SELECTION must be one of: 'score', 'random'")
        self.dynamic_alpha = getattr(extra_cfg, "DYNAMIC_ALPHA", 0.5)
        # Dynamic-prune token keep ranges (None means no upper bound in helper).
        self.dynamic_min_keep_fine = int(getattr(extra_cfg, "DYNAMIC_MIN_KEEP_FINE", 32))
        self.dynamic_max_keep_fine = getattr(extra_cfg, "DYNAMIC_MAX_KEEP_FINE", None)
        self.dynamic_min_keep_coarse = int(getattr(extra_cfg, "DYNAMIC_MIN_KEEP_COARSE", 32))
        self.dynamic_max_keep_coarse = getattr(extra_cfg, "DYNAMIC_MAX_KEEP_COARSE", None)
        configured_matching = getattr(extra_cfg, "TOKEN_MATCHING_STRATEGY", None)
        if configured_matching is None:
            configured_matching = "ot" if getattr(extra_cfg, "USE_OT_DISTILLATION", False) else "none"
        self.token_matching_strategy = str(configured_matching).strip().lower()
        if self.token_matching_strategy not in ("none", "chamfer", "ot"):
            raise ValueError("TOKEN_MATCHING_STRATEGY must be one of: 'none', 'chamfer', 'ot'")
        self.use_token_matching = self.token_matching_strategy != "none"
        self.use_ot_distillation = self.token_matching_strategy == "ot"
        # If NO_PRUNE is True, skip only the pruning / token selection steps.
        # The downstream blocks (fg_pruned_blocks, prune_recover_blocks, cross-attention,
        # and multi-grain fusion) are still kept so this behaves like a "no pruning but
        # preserve the rest of the model" baseline.
        self.no_prune = bool(getattr(extra_cfg, "NO_PRUNE", False))
        self.train_single_branch_lrb = bool(getattr(extra_cfg, "TRAIN_SINGLE_BRANCH_LRB", False))
        self.use_single_branch_loss = bool(getattr(extra_cfg, "USE_SINGLE_BRANCH_LOSS", False))
        self.single_branch_loss_weight = float(getattr(extra_cfg, "SINGLE_BRANCH_LOSS_WEIGHT", 0.1))
        self.matching_loss_weight = float(getattr(
            extra_cfg, "MATCHING_LOSS_WEIGHT", getattr(extra_cfg, "OT_LOSS_WEIGHT", 1)
        ))
        if not math.isfinite(self.matching_loss_weight) or self.matching_loss_weight < 0:
            raise ValueError("MATCHING_LOSS_WEIGHT must be finite and non-negative")
        self.ot_loss_weight = self.matching_loss_weight
        self.ot_lambda_pos = float(getattr(extra_cfg, "OT_LAMBDA_POS", 0.1))
        self.ot_blur = float(getattr(extra_cfg, "OT_BLUR", 0.05))
        if self.dynamic_max_keep_fine is not None:
            self.dynamic_max_keep_fine = int(self.dynamic_max_keep_fine)
        if self.dynamic_max_keep_coarse is not None:
            self.dynamic_max_keep_coarse = int(self.dynamic_max_keep_coarse)

        if self.use_token_matching:
            if self.token_matching_strategy == "chamfer":
                self.chamfer_distillation_loss = SymmetricChamferDistillationLoss(
                    lambda_pos=self.ot_lambda_pos,
                )
            else:
                self.ot_distillation_loss = OTDistillationLoss(
                    lambda_pos=self.ot_lambda_pos,
                    blur=self.ot_blur,
                )
            # HRB特征维度与LRB特征维度一致时，不需要投影；否则投影到统一维度再做OT。
            # 现在两者都是 EMB_DIM (1024)，所以使用 Identity
            self.ot_teacher_proj = nn.Identity()
            # 初始化last_feat_max，用于记录最后一次forward的feat_max值
            self.last_feat_max = torch.tensor(0.0)

        args = ModelArgs()
        args.st_layers ,  args.middle_layers ,  args.dim,  args.frames = 2, 2, cfg.MODEL.VIT.EMB_DIM,  self.frame_num
        args.mlp_ratio = 2
        args.use_recover = self.use_recover
        args.use_dynamic_prune = self.use_dynamic_prune
        args.no_prune = self.no_prune
        args.prune_selection = self.prune_selection
        args.use_coarse_ste_tte = bool(getattr(extra_cfg, "USE_COARSE_STE_TTE", True))
        args.dynamic_alpha = self.dynamic_alpha
        args.dynamic_min_keep = self.dynamic_min_keep_coarse
        args.dynamic_max_keep = self.dynamic_max_keep_coarse
        args.total_token_num = args.frames * self.token_num
        coarse_keep_tokens = args.total_token_num // self.coarse_prune_ratio
        if self.static_coarse_keep_tokens is not None:
            coarse_keep_tokens = self.static_coarse_keep_tokens
        args.prune_token_num = max(1, min(int(coarse_keep_tokens), args.total_token_num))
        self.prune_recover_blocks = Model(args)

        if self.use_recover:

            if self.multi_frame_fusion:
                self.fusion_mlp = nn.Sequential(
                        nn.Linear(self.token_num * self.frame_num, self.token_num),
                        nn.LayerNorm(self.token_num),
                        nn.Linear(self.token_num, self.token_num),
                        nn.LayerNorm(self.token_num),
                    )
        else:
            self.token_fusion_mlp = nn.Sequential(
                        nn.Linear(args.prune_token_num, self.token_num),
                        nn.LayerNorm(self.token_num),
                        nn.Linear(self.token_num, self.token_num),
                        nn.LayerNorm(self.token_num),
                    )

        self.multi_grain_fusion = nn.ModuleList([
            CrossBlock(cfg.MODEL.VIT.EMB_DIM, cfg.MODEL.VIT.EMB_DIM, 8, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                       drop_path=0.04, )
            for i in range(1)])
        self.multi_grain_norm = nn.LayerNorm(cfg.MODEL.VIT.EMB_DIM)
        self.inference_single_branch_lrb = bool(getattr(extra_cfg, "INFERENCE_SINGLE_BRANCH_LRB", True))

        if self.final_fusion_backbone:
            self.out_backbone_fusion = CHAIN_RSB_BLOCKS(cfg.MODEL.VIT.DECONV2_OUT_CHN+cfg.MODEL.HEAD.DECONV2_OUT_CHN,
                                                        cfg.MODEL.HEAD.FINAL_CONV_IN_CHN , 2)
        # 不拼接时，final_layer直接接收up_sample输出的128通道，无需适配

        self.up_sample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=cfg.MODEL.HEAD.DECONV1_IN_CHN,
                                   out_channels=cfg.MODEL.HEAD.DECONV1_OUT_CHN, kernel_size=4, stride=2, padding=1,
                                   output_padding=0, bias=False),
                nn.BatchNorm2d(cfg.MODEL.HEAD.DECONV1_OUT_CHN),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=cfg.MODEL.HEAD.DECONV2_IN_CHN,
                                   out_channels=cfg.MODEL.HEAD.DECONV2_OUT_CHN, kernel_size=4, stride=2, padding=1,
                                   output_padding=0, bias=False),
                nn.BatchNorm2d(cfg.MODEL.HEAD.DECONV2_OUT_CHN),
                nn.ReLU(inplace=True),

            )
        self.final_layer = nn.Conv2d(in_channels=cfg.MODEL.HEAD.FINAL_CONV_IN_CHN, out_channels=cfg.MODEL.HEAD.FINAL_CONV_OUT_CHN,
                          kernel_size=1, stride=1, padding=0)

    def _build_regular_grid_pos(self, batch_size, token_h, token_w, device, frame_idx=0):
        """构建规则网格token位置，格式为(x, y, t)，并归一化到[0,1]。"""
        yy, xx = torch.meshgrid(
            torch.arange(token_h, device=device),
            torch.arange(token_w, device=device),
            indexing='ij'
        )
        xx = xx.reshape(-1).float() / max(1, (token_w - 1))
        yy = yy.reshape(-1).float() / max(1, (token_h - 1))
        tt = torch.full_like(xx, float(frame_idx))
        pos = torch.stack((xx, yy, tt), dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        return pos

    def _build_pruned_grid_pos(self, keep_index, token_h, token_w, frame_idx=0):
        """根据剪枝后保留的index恢复token位置，保证与teacher token顺序一致。"""
        bsz, _ = keep_index.shape
        device = keep_index.device
        yy, xx = torch.meshgrid(
            torch.arange(token_h, device=device),
            torch.arange(token_w, device=device),
            indexing='ij'
        )
        base_x = (xx.reshape(-1).float() / max(1, (token_w - 1)))
        base_y = (yy.reshape(-1).float() / max(1, (token_h - 1)))

        gather_x = base_x.unsqueeze(0).expand(bsz, -1).gather(1, keep_index)
        gather_y = base_y.unsqueeze(0).expand(bsz, -1).gather(1, keep_index)
        gather_t = torch.full_like(gather_x, float(frame_idx))
        return torch.stack((gather_x, gather_y, gather_t), dim=-1)

    def _compute_ot_distill_loss(self, teacher_tokens, student_tokens, teacher_pos, student_pos, teacher_mask=None, student_mask=None):
        return self.ot_distillation_loss(
            teacher_tokens,
            student_tokens,
            teacher_pos=teacher_pos,
            student_pos=student_pos,
            teacher_mask=teacher_mask,
            student_mask=student_mask,
        )

    def _compute_matching_loss(self, teacher_tokens, student_tokens, teacher_pos, student_pos, teacher_mask=None, student_mask=None):
        if self.token_matching_strategy == "chamfer":
            return self.chamfer_distillation_loss(
                teacher_tokens,
                student_tokens,
                teacher_pos=teacher_pos,
                student_pos=student_pos,
                teacher_mask=teacher_mask,
                student_mask=student_mask,
            )
        if self.token_matching_strategy == "ot":
            return self._compute_ot_distill_loss(
                teacher_tokens,
                student_tokens,
                teacher_pos,
                student_pos,
                teacher_mask=teacher_mask,
                student_mask=student_mask,
            )
        return teacher_tokens.new_zeros(()), teacher_tokens.new_zeros(()).detach()

    def forward(self, input ,  **kwargs):
        collect_prune_vis = bool(kwargs.get("collect_prune_vis", False))
        force_full_vis = bool(kwargs.get("force_full_vis", False))
        self.prune_vis = None
        self.fine_prune_vis = None
        self.coarse_prune_vis = None

        true_batchsize , channel_num  = input.shape[0] , input.shape[1]//self.frame_num
        # print(input.shape)
        # print(true_batchsize , channel_num , dim)

        img_cat_batch= torch.cat(input.split(channel_num, dim=1), dim=0)

        #vit_backbone forward->outputs are two parts: feature and heatmaps
        htms_backbone_out_cat_batch , feats_backbone_out_cat_batch , feats_up1_backbone_out_cat_batch , feats_up2_backbone_out_cat_batch , \
                                                                                        = self.backbone(img_cat_batch)

        feat_up2_backbone_out_curr = feats_up2_backbone_out_cat_batch.split(true_batchsize, dim=0)[0]

        htms_backbone_list = list(htms_backbone_out_cat_batch.split(true_batchsize, dim=0))
        htms_curr = htms_backbone_list[0]
        run_single_lrb = self.train_single_branch_lrb or (
            (not self.training) and self.inference_single_branch_lrb and not force_full_vis
        )

        fine_total_tokens = 0
        fine_keep_tokens = 0
        feat_2x = None
        fine_keep_index = None
        fine_token_mask = None
        fine_h = 0
        fine_w = 0
        if not run_single_lrb:
            # 细粒度特征剪枝（可被 NO_PRUNE 短路）
            # 细粒度分支：只处理关键帧（中间帧）
            # feats_backbone_out_cat_batch: (b * frame_num, frames_tokens, 1024)
            # 每帧有 frames_tokens=192 个 token (16×12)

            # 使用预定义的 map_size_up_1x 作为每帧的 H, W
            # map_size_up_1x = (H, W) = (IMAGE_SIZE[1]//PATCH_SIZE, IMAGE_SIZE[0]//PATCH_SIZE)
            Hp, Wp = self.map_size_up_1x[0], self.map_size_up_1x[1]
            b = true_batchsize

            # 只提取关键帧（中间帧，索引为 frame_num//2）
            key_frame_idx = 0  # 当前帧 t
            feats_backbone_out_keyframe = feats_backbone_out_cat_batch.split(b, dim=0)[key_frame_idx]
            _, _, feat_dim = feats_backbone_out_keyframe.shape

            # reshape: (b, Hp*Wp, C) -> (b, C, Hp, Wp)
            feats_backbone_out_curr = feats_backbone_out_keyframe.reshape(-1, Hp, Wp, feat_dim).permute(0, 3, 1, 2).contiguous()

            # 反卷积上采样4倍: (b, 1024, Hp, Wp) → (b, 1024, Hp*2, Wp*2)
            feat_2x = self.fine_grain_deconv(feats_backbone_out_curr)

            # reshape为token格式: (b, 1024, H*2, W*2) → (b, H*2*W*2, 1024)
            _, _, fine_h, fine_w = feat_2x.shape
            fine_total_tokens = fine_h * fine_w
            feat_2x = feat_2x.reshape(b, -1, fine_h * fine_w).permute(0, 2, 1).contiguous()  # (b, 1024, H*2*W*2) → (b, H*2*W*2, 1024)

            if self.no_prune:
                # Bypass fine-grain token selection only: keep all tokens, but still
                # run the same fine-grain transformer blocks and normalization.
                batch_ind = torch.arange(b, device=feat_2x.device).unsqueeze(-1)
                fine_keep_index = torch.arange(fine_total_tokens, device=feat_2x.device).unsqueeze(0).expand(b, fine_total_tokens)
                fine_token_mask = torch.ones(b, fine_total_tokens, device=feat_2x.device, dtype=torch.bool)
                fine_keep_tokens = fine_total_tokens
                for blk in self.fg_pruned_blocks:
                    feat_2x = blk(feat_2x, key_padding_mask=fine_token_mask)
                feat_2x = self.fg_pruned_norm(feat_2x)
            else:
                if self.use_dynamic_prune:
                    # 动态剪枝：用自适应阈值替代固定 cluster_num（细粒度分支）
                    if collect_prune_vis:
                        index, idx_cluster, fine_token_mask, prune_debug = cluster_dpc_knn_dynamic(
                            feat_2x,
                            alpha=self.dynamic_alpha,
                            min_keep=self.dynamic_min_keep_fine,
                            max_keep=self.dynamic_max_keep_fine,
                            pad_to_batch_max=True,
                            return_mask=True,
                            return_debug=True,
                        )
                    else:
                        index, idx_cluster, fine_token_mask = cluster_dpc_knn_dynamic(
                            feat_2x,
                            alpha=self.dynamic_alpha,
                            min_keep=self.dynamic_min_keep_fine,
                            max_keep=self.dynamic_max_keep_fine,
                            pad_to_batch_max=True,
                            return_mask=True,
                        )
                        prune_debug = None
                else:
                    pruned_num = fine_h * fine_w // self.fine_grain_ratio
                    if self.static_fine_keep_tokens is not None:
                        pruned_num = self.static_fine_keep_tokens
                    pruned_num = max(1, min(int(pruned_num), fine_total_tokens))
                    index, idx_cluster = cluster_dpc_knn(feat_2x, pruned_num, 2)
                    index, _ = torch.sort(index)
                    fine_token_mask = torch.ones(b, index.shape[1], device=feat_2x.device, dtype=torch.bool)
                    prune_debug = None

                batch_ind = torch.arange(b, device=feat_2x.device).unsqueeze(-1)
                feat_2x = feat_2x[batch_ind, index]
                feat_2x = feat_2x * fine_token_mask.unsqueeze(-1).to(dtype=feat_2x.dtype)
                fine_keep_index = index
                fine_keep_tokens = float(fine_token_mask.sum(dim=1).float().mean().item())

                if collect_prune_vis and prune_debug is not None:
                    fine_prune_vis = {
                        "branch": "fine",
                        "score": prune_debug["score"].detach().cpu(),
                        "threshold": prune_debug["threshold"].detach().cpu(),
                        "candidate_keep_raw": prune_debug["candidate_keep_raw"].detach().cpu(),
                        "keep_counts": prune_debug["keep_counts"].detach().cpu(),
                        "candidate_keep_clipped": prune_debug["candidate_keep_clipped"].detach().cpu(),
                        "score_mu": prune_debug["score_mu"].detach().cpu(),
                        "score_sigma": prune_debug["score_sigma"].detach().cpu(),
                        "final_cluster_num": int(prune_debug["final_cluster_num"].detach().cpu().item()),
                        "valid_mask": fine_token_mask.detach().cpu(),
                        "clip_low_mask": prune_debug["clip_low_mask"].detach().cpu(),
                        "clip_high_mask": prune_debug["clip_high_mask"].detach().cpu(),
                        "keep_index": fine_keep_index.detach().cpu(),
                        "grid_h": int(fine_h),
                        "grid_w": int(fine_w),
                        "key_frame_idx": int(key_frame_idx),
                        "total_tokens": int(fine_total_tokens),
                    }
                    self.fine_prune_vis = fine_prune_vis
                    self.prune_vis = fine_prune_vis

                for blk in self.fg_pruned_blocks:
                    feat_2x = blk(feat_2x, key_padding_mask=fine_token_mask)
                feat_2x = self.fg_pruned_norm(feat_2x)
                feat_2x = feat_2x * fine_token_mask.unsqueeze(-1).to(dtype=feat_2x.dtype)

        #Coarse-grained multi-frame pruning and recovery
        feats_backbone_out_list = list(feats_backbone_out_cat_batch.split(true_batchsize, dim=0))
        coarse_redundancy_stats = None
        if collect_prune_vis:
            coarse_prefusion_tokens = torch.stack(feats_backbone_out_list, dim=1)
            coarse_redundancy_stats = compute_temporal_redundancy_stats(
                coarse_prefusion_tokens,
                stage="backbone_tokens_pre_pos_pre_tte",
                current_frame_index=0,
            )
        if self.add_pos_emb:
            for i in range(self.frame_num):
                feats_backbone_out_list[i] = feats_backbone_out_list[i] + self.pos_emb_list[i]

        feats_backbone_out_cat_token = torch.cat(feats_backbone_out_list,dim=1)
        coarse_total_tokens = int(feats_backbone_out_cat_token.shape[1])
        teacher_tokens_current = None
        # NO_PRUNE only disables token selection inside prune_recover_blocks.
        # The module itself still runs so its cross-attention / recovery path is preserved.
        self.prune_recover_blocks.collect_prune_vis = collect_prune_vis
        feats_backbone_out_cat_token = self.prune_recover_blocks(feats_backbone_out_cat_token)
        coarse_prune_vis = getattr(self.prune_recover_blocks, "prune_vis", None)
        if collect_prune_vis and coarse_prune_vis is not None:
            coarse_prune_vis = dict(coarse_prune_vis)
            coarse_prune_vis.update({
                "branch": "coarse",
                "grid_h": int(self.map_size_up_1x[0]),
                "grid_w": int(self.map_size_up_1x[1]),
                "key_frame_idx": 0,
            })
            if coarse_redundancy_stats is not None:
                coarse_prune_vis.update({
                    "temporal_redundancy_adj": coarse_redundancy_stats["temporal_redundancy_adj"].detach().cpu(),
                    "temporal_redundancy_center": coarse_redundancy_stats["temporal_redundancy_center"].detach().cpu(),
                    "temporal_redundancy_adj_chrono_prefusion": coarse_redundancy_stats["temporal_redundancy_adj_chrono_prefusion"].detach().cpu(),
                    "temporal_redundancy_current_support_prefusion": coarse_redundancy_stats["temporal_redundancy_current_support_prefusion"].detach().cpu(),
                    "temporal_order": coarse_redundancy_stats["temporal_order"],
                    "temporal_redundancy_stage": coarse_redundancy_stats["temporal_redundancy_stage"],
                    "current_frame_index": coarse_redundancy_stats["current_frame_index"],
                })
            self.coarse_prune_vis = coarse_prune_vis
            if self.prune_vis is None:
                self.prune_vis = coarse_prune_vis

        # 获取粗粒度分支实际剪枝后的token数量
        if self.no_prune:
            coarse_keep_tokens = coarse_total_tokens
        else:
            coarse_keep_counts = getattr(self.prune_recover_blocks, "prune_keep_counts", None)
            if coarse_keep_counts is not None:
                coarse_keep_tokens = float(coarse_keep_counts.float().mean().item())
            else:
                # prune_recover_blocks 内部会更新 prune_token_num（动态剪枝时是 padding 后的 Kmax）
                coarse_keep_tokens = int(getattr(self.prune_recover_blocks, "prune_token_num", coarse_total_tokens))

        self.token_keep_stats = {
            "fine_total_tokens": fine_total_tokens,
            "fine_keep_tokens": fine_keep_tokens,
            "fine_keep_ratio": float(fine_keep_tokens / max(1, fine_total_tokens)),
            "coarse_total_tokens": coarse_total_tokens,
            "coarse_keep_tokens": coarse_keep_tokens,
            "coarse_keep_ratio": float(coarse_keep_tokens / max(1, coarse_total_tokens)),
        }

        self.token_keep_history["fine_keep_tokens"].append(fine_keep_tokens)
        self.token_keep_history["coarse_keep_tokens"].append(coarse_keep_tokens)
        self.token_keep_history["fine_keep_ratio"].append(self.token_keep_stats["fine_keep_ratio"])
        self.token_keep_history["coarse_keep_ratio"].append(self.token_keep_stats["coarse_keep_ratio"])

        if self.use_recover:
            if self.multi_frame_fusion:

                out = self.fusion_mlp(feats_backbone_out_cat_token.permute(0,2,1).contiguous() )

            else:
                out =  feats_backbone_out_cat_token.split(self.token_num, dim=1)[0].permute(0,2,1).contiguous()
        else:
            out = self.token_fusion_mlp(feats_backbone_out_cat_token.permute(0,2,1).contiguous())

        ot_loss = out.new_zeros(())
        feat_max = torch.tensor(0.0, device=out.device)
        if self.use_token_matching and (not run_single_lrb):
            # Token matching uses HRB(teacher)=细粒度分支，LRB(student)=粗粒度主分支。
            teacher_tokens_current = self.ot_teacher_proj(feat_2x)
            student_tokens_current = out.permute(0, 2, 1)
            teacher_pos = self._build_pruned_grid_pos(
                fine_keep_index,
                token_h=fine_h,
                token_w=fine_w,
                frame_idx=0,
            )
            student_pos = self._build_regular_grid_pos(
                batch_size=true_batchsize,
                token_h=self.map_size_up_1x[0],
                token_w=self.map_size_up_1x[1],
                device=student_tokens_current.device,
                frame_idx=0,
            )
            ot_loss, feat_max = self._compute_matching_loss(
                teacher_tokens_current,
                student_tokens_current,
                teacher_pos,
                student_pos,
                teacher_mask=fine_token_mask,
            )
            ot_loss = ot_loss * self.matching_loss_weight

        # --- 单分支输出（粗粒度分支不经 cross attention，直接走输出头产生的结果）---
        out_single_branch = None
        need_single_branch = (not run_single_lrb) and self.use_single_branch_loss and self.training
        if need_single_branch:
            out_single = self.multi_grain_norm(out.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
            out_single_branch = self.up_sample(
                out_single.reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1]).contiguous()
            )
            if self.final_fusion_backbone:
                out_backbone_cat_single = torch.cat((out_single_branch, feat_up2_backbone_out_curr), dim=1)
                out_single_branch = self.out_backbone_fusion(out_backbone_cat_single)
            out_single_branch = self.final_layer(out_single_branch)

        if run_single_lrb:
            out = self.multi_grain_norm(out.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        else:
            for blk in self.multi_grain_fusion:
                out = blk(out.permute(0, 2, 1).contiguous(), feat_2x, feat_mask=fine_token_mask)
            out = self.multi_grain_norm(out).permute(0,2,1).contiguous()

        #head
        out = self.up_sample(out.reshape(true_batchsize, -1, self.map_size_up_1x[0], self.map_size_up_1x[1]).contiguous())

        if self.final_fusion_backbone:
            out_backbone_cat = torch.cat((out , feat_up2_backbone_out_curr), dim=1)
            out = self.out_backbone_fusion(out_backbone_cat)
        # 不拼接时，out已经是128通道，直接送入final_layer
        out = self.final_layer(out)

        prev_ref_idx = 1 if len(htms_backbone_list) > 1 else 0
        next_ref_idx = 1 + max(1, (self.frame_num - 1) // 2)
        if next_ref_idx >= len(htms_backbone_list):
            next_ref_idx = len(htms_backbone_list) - 1
        htms_supp_A = htms_backbone_list[prev_ref_idx]
        htms_supp_B = htms_backbone_list[next_ref_idx]
        self.htms_dict = {
            "diff_f": htms_curr - htms_supp_A,
            "diff_b": htms_curr - htms_supp_B,
            "htms_out": out,
        }

        if run_single_lrb:
            if self.use_token_matching:
                self.last_feat_max = torch.tensor(0.0, device=out.device)
                return out, htms_curr, out.new_zeros(())
            else:
                return out, htms_curr
        else:
            if self.use_token_matching:
                self.last_feat_max = feat_max
                if need_single_branch:
                    # 4元素：(out, htms, matching_loss, out_single_branch)
                    return out, htms_curr, ot_loss, out_single_branch
                else:
                    # 3元素：(out, htms, matching_loss)
                    return out, htms_curr, ot_loss
            else:
                if need_single_branch:
                    # 4元素：(out, htms, 0.0, out_single_branch) — 第3位固定为ot_loss占位
                    return out, htms_curr, out.new_zeros(()), out_single_branch
                else:
                    # 2元素：(out, htms)
                    return out, htms_curr

    def init_weights(self):
        logger = logging.getLogger(__name__)

        for module_name, module in self.named_modules():
            # Skip backbone here. Backbone keeps its own pretrained loading logic.
            if module_name.startswith("backbone"):
                continue

            if isinstance(module, nn.Conv2d):
                # Conv layers in OTAPose heads/fusion are mostly followed by ReLU-family paths.
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                # Deconv upsampling path is followed by BN + ReLU.
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Transformer-style projections/MLPs use Xavier for stable variance propagation.
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            else:
                # Fallback for uncommon modules: only touch direct params of current module.
                for name, param in module.named_parameters(recurse=False):
                    if name == 'bias' and param is not None:
                        nn.init.constant_(param, 0)
                    elif name == 'weight' and param is not None and param.dim() >= 2:
                        nn.init.xavier_uniform_(param)

        self.backbone.init_weights()
        if self.freeze_backbone_weights:
            self.backbone.freeze_weight()

    def reset_token_keep_history(self):
        self.token_keep_history = {
            "fine_keep_tokens": [],
            "coarse_keep_tokens": [],
            "fine_keep_ratio": [],
            "coarse_keep_ratio": [],
        }
        self.token_keep_stats = dict()

    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):

        hyper_parameters_setting = "OTA-Pose"
        return hyper_parameters_setting
