#!/usr/bin/python
# -*- coding:utf8 -*-
"""
DiffusionPose Tracking Function：训练 + 按视频顺序验证/推理，维护轨迹池、首帧赋 id、后续帧匹配。

训练：与 3_frame 类似，接收 (prev, cur, cur) 数据，只监督当前帧。
验证：按视频顺序处理，维护轨迹池，第一帧所有人给新 id，后续帧用 bbox IoU 匹配。
"""
import time
import torch
import numpy as np
import os.path as osp
import logging
from collections import defaultdict
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
from ..evaludate import accuracy, pck_accuracy, pck_accuracy_origin_image
from engine.core import CORE_FUNCTION_REGISTRY, BaseFunction, AverageMeter
import torch
import torch.nn.functional as F

from engine.defaults import VAL_PHASE, TEST_PHASE, TRAIN_PHASE
from datasets.process import get_final_preds
from datasets.transforms import reverse_transforms
from datasets.process import get_final_preds, dark_get_final_preds
from utils.utils_bbox import cs2box, box2cs
from utils.utils_folder import create_folder
from utils.utils_image_tensor import tensor2im
from engine.core.DiffusionPose.alphaposeutil.utils.bbox import bbox_iou, bbox_xywh_to_xyxy

from torch.nn.functional import kl_div, softmax
from tabulate import tabulate

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
    logging.warning("scipy.optimize.linear_sum_assignment not available, using greedy matching")


# 三状态轨迹（模仿 AlphaPose / DeepSORT）
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


# PoseTrack 标注中间约 30 帧，丢失容忍与之一致；可从 cfg.VAL.TRACK_MAX_AGE 覆盖
TRACK_MAX_AGE_DEFAULT = 30


def cnt_grad(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm().item()
    return total_norm


@CORE_FUNCTION_REGISTRY.register()
class Function_DiffusionPose_Tracking(BaseFunction):
    """
    DiffusionPose Tracking Function：支持训练和按视频顺序的验证/推理。
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if "criterion" in kwargs.keys():
            self.criterion = kwargs["criterion"]
        if "tb_log_dir" in kwargs.keys():
            self.tb_log_dir = kwargs["tb_log_dir"]
        if "writer_dict" in kwargs.keys():
            self.writer_dict = kwargs["writer_dict"]

        self.PE_Name = kwargs.get("PE_Name", "DiffusionPOSE_Tracking")
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0
        self.softmax = torch.nn.Softmax(dim=1)
        self.DataSetName = str(self.cfg.DATASET.NAME).upper()
        self.mlp = torch.nn.Linear(cfg.MODEL.VIT.EMB_DIM, cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0]).cuda()

    def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, **kwargs):
        """
        训练方法：与 Function_DiffusionPose_3_frame 类似，但接收 8 个返回值（包括 prev_heatmaps）。
        目前只使用当前帧的 target_heatmaps 做 loss，prev_heatmaps 暂不使用（为后续 ST-MRF 预留）。
        """
        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]
        logger = logging.getLogger(__name__)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        grad = AverageMeter()
        MI_LOSS_1 = AverageMeter()
        MI_LOSS_2 = AverageMeter()
        MI_FEAT = AverageMeter()
        EDGE_LOSS = AverageMeter()

        model.train()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        end = time.time()

        for iter_step in range(self.max_iter_num):
            # 接收 8 个返回值：input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight,
            #                   prev_heatmaps, next_heatmaps, meta
            input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, \
                prev_heatmaps, next_heatmaps, meta = next(self.dataloader_iter)

            self._before_train_iter(input_x)

            data_time.update(time.time() - end)

            margin_left = meta["margin_left"]
            margin_right = meta["margin_right"]
            margin = torch.stack([margin_left, margin_right], dim=1).cuda()
            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)

            concat_input = torch.cat((input_x, input_prev, input_next), 1).cuda()
            outputs, htms_backbone, feat_curr, fused_feature, kpt_tokens_curr = model(
                concat_input, target=target_heatmaps, margin=margin)

            if isinstance(outputs, list) or isinstance(outputs, tuple):
                pred_heatmaps = outputs[0]
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
                for pred_heatmaps in outputs[1:]:
                    loss += self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
            else:
                pred_heatmaps = outputs
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)

            # MI loss（与 3_frame 保持一致）
            mi_loss_1 = self.feat_label_mi_estimation_V1(target_heatmaps, fused_feature, model) \
                        - self.feat_feat_mi_estimation(fused_feature, feat_curr)
            mi_loss_2 = self.feat_label_mi_estimation_keypoint_gt(target_heatmaps, kpt_tokens_curr)
            alpha = 1
            beta = 0.1
            gamma = 0.001
            mi_feat = (-1) * beta * mi_loss_1 + (-1) * gamma * mi_loss_2
            mi_loss = alpha * mi_feat
            loss += mi_loss

            # Edge loss
            edge_loss = self.edge_potential_loss(pred_heatmaps, target_heatmaps, lambda_length=1.0, lambda_angle=1.0)
            edge_loss = 0.0001 * edge_loss
            loss += edge_loss

            # 注意：目前不使用 prev_heatmaps 做 loss，为后续 ST-MRF 预留接口
            # 如果后续要加 ST-MRF，可以在这里添加：
            # if meta.get('labeled_prev', 0) == 1:
            #     prev_target = torch.from_numpy(prev_heatmaps).cuda()
            #     prev_loss = self.criterion(prev_head_output, prev_target, ...)
            #     loss += prev_loss

            optimizer.zero_grad()
            loss.backward()
            grad.update(cnt_grad(model))
            optimizer.step()

            losses.update(loss.item(), input_x.size(0))
            MI_FEAT.update(mi_feat.item(), input_x.size(0))
            EDGE_LOSS.update(edge_loss.item(), input_x.size(0))

            _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()
            if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= self.max_iter_num - 1:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'MI_FEAT {MI_FEAT.val:.5f} ({MI_FEAT.avg:.5f})\t' \
                      'Edge_loss {EDGE_LOSS.val:.5f} ({EDGE_LOSS.avg:.5f})\t' \
                      'Grad {grad.val:.3f} ({grad.avg:.3f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                    epoch, iter_step, self.max_iter_num, batch_time=batch_time,
                    speed=input_x.size(0) / batch_time.val,
                    data_time=data_time, grad=grad, loss=losses, acc=acc, MI_FEAT=MI_FEAT, EDGE_LOSS=EDGE_LOSS)

                logger.info(msg)

            self.tb_writer.add_scalar('train_loss', losses.val, self.global_steps)
            self.tb_writer.add_scalar('train_acc', acc.val, self.global_steps)
            self.global_steps += 1

        tb_writer_dict["global_steps"] = self.global_steps

    def eval(self, model, dataloader, tb_writer_dict, **kwargs):
        """
        验证方法：按视频顺序处理，维护轨迹池，第一帧所有人给新 id，后续帧匹配。
        
        核心流程：
        1. 从 dataset 获取所有视频列表，按视频分组
        2. 对每个视频，按 frame_id 排序
        3. 逐帧处理：
           - 第一帧：所有人分配新 track_id，加入轨迹池
           - 后续帧：用 bbox IoU 匹配，匹配上沿用 track_id，未匹配分配新 id
        4. 收集所有预测和 track_id，调用 dataset.evaluate_tracking()
        """
        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time = AverageMeter()
        data_time = AverageMeter()
        acc = AverageMeter()

        phase = kwargs.get("phase", VAL_PHASE)
        epoch = kwargs.get("epoch", "specified_model")
        model.eval()

        dataset = dataloader.dataset
        result_output_dir, vis_output_dir = self.vis_setup(logger, phase, epoch)

        # ====================================================================
        # 按视频组织数据（不入 dataloader 的随机顺序）
        # ====================================================================
        logger.info("=> Organizing data by video for tracking evaluation...")
        video_to_samples = self._organize_data_by_video(dataset)
        logger.info(f"=> Found {len(video_to_samples)} videos")

        # 存储所有预测结果（用于最终评估）
        all_preds_list = []  # 每个元素是 (image_path, preds, boxes, track_ids)
        all_track_ids_dict = {}  # key: image_path, value: list of track_ids

        # ====================================================================
        # 对每个视频按帧顺序处理
        # ====================================================================
        total_samples = 0
        with torch.no_grad():
            for video_idx, (video_name, sample_indices) in enumerate(video_to_samples.items()):
                logger.info(f"=> Processing video {video_idx + 1}/{len(video_to_samples)}: {video_name}")

                # 获取该视频的所有样本，按 frame_id 排序
                video_samples = []
                for idx in sample_indices:
                    sample = dataset.data[idx]
                    video_samples.append((idx, sample))

                # 按 frame_id 排序
                video_samples_sorted = sorted(video_samples, key=lambda x: (x[1]['frame_id'], x[0]))

                # 三状态轨迹池（模仿 AlphaPose）
                # tracked_stracks: 本帧或此前匹配上的轨迹
                # lost_stracks: 暂时丢失，在 max_age 帧内仍参与匹配
                # 每条轨迹含: track_id, last_bbox, last_pose, last_frame_id, last_center, last_scale, last_crop, state, end_frame
                tracked_stracks = []
                lost_stracks = []
                next_track_id = 0
                max_age = getattr(self.cfg.VAL, 'TRACK_MAX_AGE', TRACK_MAX_AGE_DEFAULT)

                # 当前帧的所有检测（用于批量匹配）
                current_frame_detections = []
                current_frame_id = None

                # 逐样本处理（每个样本是一个人）
                for sample_idx_in_video, (dataset_idx, sample) in enumerate(video_samples_sorted):
                    frame_id = sample['frame_id']
                    image_path = sample['image']
                    center = sample['center']
                    scale = sample['scale']
                    box = sample.get('box', None)  # [x, y, w, h]

                    # 如果是新的一帧，先处理上一帧的所有检测（批量匹配）
                    if current_frame_id is not None and frame_id != current_frame_id:
                        # 处理上一帧的所有检测；返回更新后的 tracked / lost 与 next_track_id
                        tracked_stracks, lost_stracks, next_track_id = self._process_frame_detections(
                            current_frame_detections, tracked_stracks, lost_stracks, next_track_id,
                            current_frame_id, max_age, all_preds_list, all_track_ids_dict, logger)
                        current_frame_detections = []

                    current_frame_id = frame_id

                    # 获取该样本的数据
                    end_data = time.time()
                    input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, \
                        prev_heatmaps, next_heatmaps, meta = dataset.__getitem__(dataset_idx)
                    data_time.update(time.time() - end_data)

                    # 确定前一帧的 crop（用于模型输入）
                    # 如果是第一帧，prev = cur；否则用匹配到的轨迹的上一帧 crop
                    is_first_frame = (sample_idx_in_video == 0) or (current_frame_id != video_samples_sorted[0][1]['frame_id'])
                    
                    if is_first_frame and len(tracked_stracks) == 0 and len(lost_stracks) == 0:
                        # 第一帧第一个人：prev = cur
                        prev_crop_for_model = input_x.unsqueeze(0)  # [1, C, H, W]
                    else:
                        # 后续帧或同一帧的后续人：用 dataset 返回的 input_prev（已经是前一帧的 crop）
                        prev_crop_for_model = input_prev.unsqueeze(0)

                    # 构造模型输入：prev + cur + next（next=cur）
                    input_batch = torch.cat([
                        prev_crop_for_model,
                        input_x.unsqueeze(0),
                        input_x.unsqueeze(0)  # next = cur
                    ], dim=1).cuda()  # [1, 9, H, W]

                    # 获取 margin（从 meta 中）
                    margin_left = meta.get('margin_left', 1)
                    margin_right = meta.get('margin_right', 1)
                    margin = torch.tensor([[margin_left, margin_right]], dtype=torch.long).cuda()

                    # 模型推理
                    end_infer = time.time()
                    outputs, htms_backbone, feat_curr, fused_feature, kpt_tokens_curr = model(
                        input_batch, margin=margin)
                    batch_time.update(time.time() - end_infer)

                    if isinstance(outputs, list) or isinstance(outputs, tuple):
                        pred_heatmaps = outputs[-1]
                    else:
                        pred_heatmaps = outputs

                    # Flip test（如果启用）
                    FLIP_TEST = self.cfg.VAL.FLIP_VAL
                    if FLIP_TEST:
                        input_batch_flipped = input_batch.flip(3)
                        pred_heatmaps_flipped, _, _, _, _ = model(input_batch_flipped, margin=margin)
                        from datasets.process.pose_process import flip_back
                        pred_heatmaps_flipped = flip_back(
                            pred_heatmaps_flipped.cpu().numpy(), dataset.flip_pairs)
                        pred_heatmaps_flipped = torch.from_numpy(pred_heatmaps_flipped.copy()).cuda()
                        pred_heatmaps_flipped[:, :, :, 1:] = pred_heatmaps_flipped.clone()[:, :, :, 0:-1]
                        pred_heatmaps = (pred_heatmaps + pred_heatmaps_flipped) * 0.5

                    # 解码得到关节点坐标
                    center_np = center.reshape(1, 2) if isinstance(center, np.ndarray) else np.array([[center[0], center[1]]])
                    scale_np = scale.reshape(1, 2) if isinstance(scale, np.ndarray) else np.array([[scale[0], scale[1]]])
                    preds, maxvals = dark_get_final_preds(
                        pred_heatmaps.cpu().numpy(), center_np, scale_np)

                    # 计算 bbox（用于匹配）
                    if box is None:
                        bbox_xyxy = cs2box(center, scale, pattern="xyxy")
                        bbox_xywh = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]]
                    else:
                        bbox_xywh = list(box) if isinstance(box, (list, tuple, np.ndarray)) else box  # [x, y, w, h]

                    # 存储当前检测（稍后批量匹配）
                    current_frame_detections.append({
                        'dataset_idx': dataset_idx,
                        'image_path': image_path,
                        'preds': preds[0],
                        'maxvals': maxvals[0],
                        'bbox': bbox_xywh,
                        'center': center,
                        'scale': scale,
                        'frame_id': frame_id,
                        'input_x': input_x,
                        'input_prev': input_prev,
                        'meta': meta,
                    })

                # 处理最后一帧的所有检测
                if len(current_frame_detections) > 0:
                    tracked_stracks, lost_stracks, next_track_id = self._process_frame_detections(
                        current_frame_detections, tracked_stracks, lost_stracks, next_track_id,
                        current_frame_id, max_age, all_preds_list, all_track_ids_dict, logger)

                total_samples += len(video_samples_sorted)
                logger.info(f"  Video {video_name}: Processed {len(video_samples_sorted)} samples, "
                           f"final tracked: {len(tracked_stracks)}, lost: {len(lost_stracks)}")

        # ====================================================================
        # 整理结果，调用 evaluate_tracking
        # ====================================================================
        logger.info(f"=> Total processed {total_samples} samples across {len(video_to_samples)} videos")

        # 构建 evaluate_tracking 需要的格式
        # all_preds: (N, num_joints, 3)
        # all_boxes: (N, 6)
        # img_path: dict, key=image_path, value=list of indices
        # track_ids: dict, key=image_path, value=list of track_ids

        num_samples = len(all_preds_list)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=float)
        all_boxes = np.zeros((num_samples, 6))
        img_path_dict = {}
        track_ids_dict = {}

        for idx, item in enumerate(all_preds_list):
            all_preds[idx, :, 0:2] = item['preds']
            all_preds[idx, :, 2:3] = item['maxvals']
            all_boxes[idx, 0:2] = item['center'][:2] if isinstance(item['center'], np.ndarray) else [item['center'][0], item['center'][1]]
            all_boxes[idx, 2:4] = item['scale'][:2] if isinstance(item['scale'], np.ndarray) else [item['scale'][0], item['scale'][1]]
            all_boxes[idx, 4] = np.prod(item['scale'][:2] * 200) if isinstance(item['scale'], np.ndarray) else item['scale'][0] * item['scale'][1] * 200 * 200
            all_boxes[idx, 5] = 1.0  # score

            image_path = item['image_path']
            if image_path not in img_path_dict:
                img_path_dict[image_path] = []
            img_path_dict[image_path].append(idx)

            if image_path not in track_ids_dict:
                track_ids_dict[image_path] = []
            track_ids_dict[image_path].append(item['track_id'])

        # 调用 evaluate_tracking
        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))

        if hasattr(dataset, 'evaluate_tracking'):
            name_values, perf_indicator = dataset.evaluate_tracking(
                self.cfg, all_preds, result_output_dir, all_boxes, img_path_dict, track_ids_dict)
        else:
            logger.warning("dataset.evaluate_tracking() not found, falling back to evaluate()")
            name_values, perf_indicator = dataset.evaluate(
                self.cfg, all_preds, result_output_dir, all_boxes, img_path_dict)

        model_name = self.cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, model_name)
        else:
            self._print_name_value(name_values, model_name)

        tb_writer_dict["global_steps"] = self.global_steps

    def _organize_data_by_video(self, dataset):
        """
        从 dataset 中按视频组织数据。
        
        Returns:
            video_to_samples: dict, key=video_name, value=list of dataset indices
        """
        video_to_samples = defaultdict(list)

        for idx, sample in enumerate(dataset.data):
            # 从 image 路径提取 video_name（与数据集中的方式一致）
            image_path = sample['image']
            # 路径格式：./Dataset/PoseTrack2017/images_renamed/video_name/xxxxx.jpg
            # 提取方式：与 evaluate() 方法中的方式一致
            path_parts = image_path.split('/')
            if len(path_parts) >= 3:
                # 取倒数第二和倒数第三部分作为 video_name（与 evaluate 中的格式一致）
                video_name = path_parts[-3] + '/' + path_parts[-2]
            else:
                # fallback：用目录名
                video_name = osp.basename(osp.dirname(image_path))

            video_to_samples[video_name].append(idx)

        return dict(video_to_samples)

    def _process_frame_detections(self, detections, tracked_stracks, lost_stracks, next_track_id_start, frame_id,
                                   max_age, all_preds_list, all_track_ids_dict, logger):
        """
        处理一帧内的所有检测，三状态（Tracked / Lost / Removed）+ max_age。
        模仿 AlphaPose：pool = tracked + lost(未超时)，匹配后未匹配的 Tracked -> Lost，超时 Lost -> Removed。

        Returns:
            (new_tracked_stracks, new_lost_stracks, next_track_id)
        """
        if len(detections) == 0:
            # 本帧无检测：当前所有 Tracked 视为未匹配 -> Mark Lost；Lost 中超时的移除
            new_lost = []
            for t in tracked_stracks:
                t = dict(t)
                t['state'] = TrackState.Lost
                t['end_frame'] = frame_id
                new_lost.append(t)
            lost_stracks = [t for t in lost_stracks + new_lost if frame_id - t['end_frame'] <= max_age]
            return [], lost_stracks, next_track_id_start

        next_track_id = next_track_id_start
        is_first_frame = (len(tracked_stracks) == 0 and len(lost_stracks) == 0)

        def _make_track(det, tid, state=TrackState.Tracked):
            return {
                'track_id': tid,
                'last_bbox': det['bbox'].copy() if isinstance(det['bbox'], np.ndarray) else list(det['bbox']),
                'last_pose': det['preds'].copy(),
                'last_frame_id': frame_id,
                'last_center': det['center'].copy() if isinstance(det['center'], np.ndarray) else np.array(det['center']),
                'last_scale': det['scale'].copy() if isinstance(det['scale'], np.ndarray) else np.array(det['scale']),
                'last_crop': det['input_x'].clone(),
                'state': state,
                'end_frame': frame_id,  # 最后被看到的帧
            }

        def _write_pred(det, tid):
            all_preds_list.append({
                'image_path': det['image_path'],
                'preds': det['preds'],
                'maxvals': det['maxvals'],
                'bbox': det['bbox'],
                'center': det['center'],
                'scale': det['scale'],
                'track_id': tid,
                'frame_id': frame_id,
            })
            if det['image_path'] not in all_track_ids_dict:
                all_track_ids_dict[det['image_path']] = []
            all_track_ids_dict[det['image_path']].append(tid)

        if is_first_frame:
            # 第一帧：所有人分配新 track_id，状态 Tracked
            new_tracked = []
            for det in detections:
                track_id = next_track_id
                next_track_id += 1
                new_tracked.append(_make_track(det, track_id, TrackState.Tracked))
                _write_pred(det, track_id)
            return new_tracked, [], next_track_id

        # 未超时的 Lost 参与匹配
        lost_alive = [t for t in lost_stracks if frame_id - t['end_frame'] <= max_age]
        pool = list(tracked_stracks) + list(lost_alive)

        if len(pool) == 0:
            # 池子为空（全超时或全 Removed），全部当新轨迹
            new_tracked = []
            for det in detections:
                track_id = next_track_id
                next_track_id += 1
                new_tracked.append(_make_track(det, track_id, TrackState.Tracked))
                _write_pred(det, track_id)
            return new_tracked, [], next_track_id

        # IoU 匹配：pool vs detections
        num_tracks = len(pool)
        num_dets = len(detections)
        track_bboxes = [list(t['last_bbox']) for t in pool]
        det_bboxes = [list(d['bbox']) for d in detections]
        track_bboxes_xyxy = np.array([bbox_xywh_to_xyxy(b) for b in track_bboxes], dtype=np.float32)
        det_bboxes_xyxy = np.array([bbox_xywh_to_xyxy(b) for b in det_bboxes], dtype=np.float32)
        iou_matrix = bbox_iou(track_bboxes_xyxy, det_bboxes_xyxy, offset=1)

        matched_track_indices = []
        matched_det_indices = []
        unmatched_track_indices = list(range(num_tracks))
        unmatched_det_indices = list(range(num_dets))

        iou_threshold = 0.5
        if linear_sum_assignment is not None and num_tracks > 0 and num_dets > 0:
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] > iou_threshold:
                    matched_track_indices.append(r)
                    matched_det_indices.append(c)
                    if r in unmatched_track_indices:
                        unmatched_track_indices.remove(r)
                    if c in unmatched_det_indices:
                        unmatched_det_indices.remove(c)
        else:
            for det_idx in range(num_dets):
                if len(unmatched_track_indices) == 0:
                    break
                best_iou, best_track_idx = 0.0, None
                for track_idx in unmatched_track_indices:
                    iou = iou_matrix[track_idx, det_idx]
                    if iou > best_iou and iou > iou_threshold:
                        best_iou, best_track_idx = iou, track_idx
                if best_track_idx is not None:
                    matched_track_indices.append(best_track_idx)
                    matched_det_indices.append(det_idx)
                    unmatched_track_indices.remove(best_track_idx)
            unmatched_det_indices = [i for i in range(num_dets) if i not in matched_det_indices]

        new_tracked_stracks = []
        new_lost_stracks = []

        # 已匹配：更新轨迹，统一为 Tracked（含 re_activate）
        for track_idx, det_idx in zip(matched_track_indices, matched_det_indices):
            det = detections[det_idx]
            tr = pool[track_idx]
            track_id = tr['track_id']
            tr['last_bbox'] = det['bbox'].copy() if isinstance(det['bbox'], np.ndarray) else list(det['bbox'])
            tr['last_pose'] = det['preds'].copy()
            tr['last_frame_id'] = frame_id
            tr['last_center'] = det['center'].copy() if isinstance(det['center'], np.ndarray) else np.array(det['center'])
            tr['last_scale'] = det['scale'].copy() if isinstance(det['scale'], np.ndarray) else np.array(det['scale'])
            tr['last_crop'] = det['input_x'].clone()
            tr['state'] = TrackState.Tracked
            tr['end_frame'] = frame_id
            new_tracked_stracks.append(tr)
            _write_pred(det, track_id)

        # 未匹配的轨迹：Tracked -> Lost，Lost -> 仍 Lost（保留在 new_lost_stracks，下一帧继续参与）
        for track_idx in unmatched_track_indices:
            tr = pool[track_idx]
            tr['state'] = TrackState.Lost
            tr['end_frame'] = frame_id  # 本帧起丢失
            new_lost_stracks.append(tr)

        # 未匹配的检测：新轨迹
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            track_id = next_track_id
            next_track_id += 1
            new_tracked_stracks.append(_make_track(det, track_id, TrackState.Tracked))
            _write_pred(det, track_id)

        return new_tracked_stracks, new_lost_stracks, next_track_id

    # ====================================================================
    # 以下方法从 Function_DiffusionPose_3_frame 复制过来
    # ====================================================================

    def decode_heatmap_to_coords(self, heatmaps):
        """从热图中解码关节点的坐标。"""
        B, J, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, J, -1)
        idx = torch.argmax(heatmaps, dim=-1)
        y_coords = (idx // W + 1) / H * 1.00
        x_coords = (idx % W + 1) / W * 1.00
        coords = torch.stack([x_coords, y_coords], dim=-1)
        return coords

    def edge_potential_loss(self, pred_heatmaps, target_heatmaps, lambda_length=1.0, lambda_angle=1.0):
        """计算基于边势函数的损失。"""
        B, J, H, W = pred_heatmaps.shape
        limb_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (5, 11), (6, 8), (6, 12),
                      (7, 9), (8, 10), (11, 13), (12, 14), (13, 15), (14, 16)]
        angle_constraints = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 3, 5), (2, 4, 6), (5, 7, 9), (6, 8, 10),
                            (11, 13, 15), (12, 14, 16)]

        pred_coords = self.decode_heatmap_to_coords(pred_heatmaps)
        target_coords = self.decode_heatmap_to_coords(target_heatmaps)

        pred_limb_lengths = torch.zeros(B, len(limb_pairs)).to(pred_heatmaps.device)
        target_limb_lengths = torch.zeros(B, len(limb_pairs)).to(pred_heatmaps.device)

        for i, (j1, j2) in enumerate(limb_pairs):
            pred_limb_lengths[:, i] = torch.norm(pred_coords[:, j1] - pred_coords[:, j2], dim=-1)
            target_limb_lengths[:, i] = torch.norm(target_coords[:, j1] - target_coords[:, j2], dim=-1)

        std_length = torch.std(target_limb_lengths, dim=-1, unbiased=False).unsqueeze(-1)
        limb_length_loss = torch.exp(
            -1.0 * (pred_limb_lengths - target_limb_lengths) ** 2 / (2 * std_length ** 2 + 1e-9)
        )
        limb_length_loss = -torch.log(torch.mean(limb_length_loss, dim=-1) + 1e-9)

        def compute_angles(coords, angle_constraints):
            angles = []
            for i, j, k in angle_constraints:
                v1 = coords[:, i] - coords[:, j]
                v2 = coords[:, k] - coords[:, j]
                v1_norm = torch.norm(v1, dim=-1, keepdim=True) + 1e-6
                v2_norm = torch.norm(v2, dim=-1, keepdim=True) + 1e-6
                v1_normalized = v1 / v1_norm
                v2_normalized = v2 / v2_norm
                cos_angle = torch.sum(v1_normalized * v2_normalized, dim=-1)
                angles.append(cos_angle)
            return torch.stack(angles, dim=1)

        pred_angles = compute_angles(pred_coords, angle_constraints)
        target_angles = compute_angles(target_coords, angle_constraints)

        angle_loss = torch.mean((pred_angles - target_angles) ** 2, dim=-1) + 1e-6
        edge_loss = lambda_length * torch.mean(limb_length_loss.view(-1)) + lambda_angle * torch.mean(angle_loss.view(-1))

        return edge_loss

    def feat_label_mi_estimation_V1(self, Y, Feat, model):
        """F: [B,192,1024], Y: [B,17,64,48]"""
        batch_size = Feat.shape[0]
        temperature = 0.05
        Feat = Feat.permute(0, 2, 1).reshape(batch_size, -1, 16, 12).contiguous()

        for layer in model.backbone.deconv_layers:
            Feat = layer(Feat)
        pred = model.backbone.final_layer(Feat)

        pred = pred.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        Y = Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        mi = kl_div(input=self.softmax(pred.detach() / temperature), target=self.softmax(Y / temperature),
                    reduction='mean')
        return mi

    def feat_feat_mi_estimation(self, F1, F2):
        """F1: [B,192,1024], F2: [B,192,1024]"""
        batch_size, N = F1.shape[0], F1.shape[1]
        temperature = 0.05
        F1 = F1.reshape(batch_size * N, -1)
        F2 = F2.reshape(batch_size * N, -1)
        mi = kl_div(input=self.softmax(F1.detach() / temperature), target=self.softmax(F2 / temperature),
                    reduction='mean')
        return mi

    def feat_label_mi_estimation_keypoint_gt(self, gt, kpt_tokens):
        batch_size = gt.shape[0]
        temperature = 0.05
        pred = self.mlp(kpt_tokens)
        pred = pred.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        Y = gt.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        mi = kl_div(input=self.softmax(pred.detach() / temperature), target=self.softmax(Y / temperature),
                    reduction='mean')
        return mi

    def _before_train_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return
        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "train_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def vis_setup(self, logger, phase, epoch):
        if phase == TEST_PHASE:
            prefix_dir = "test"
        elif phase == TRAIN_PHASE:
            prefix_dir = "train"
        elif phase == VAL_PHASE:
            prefix_dir = "validate"
        else:
            prefix_dir = "inference"

        if isinstance(epoch, int):
            epoch = "model_{}".format(str(epoch))

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir,
                                   "use_gt_box" if self.cfg.VAL.USE_GT_BBOX else "use_precomputed_box")
        vis_output_dir = osp.join(output_dir_base, "vis")
        result_output_dir = osp.join(output_dir_base, "prediction_result")
        create_folder(vis_output_dir)
        create_folder(result_output_dir)
        logger.info("=> Vis Output Dir : {}".format(vis_output_dir))
        logger.info("=> Result Output Dir : {}".format(result_output_dir))

        if phase == VAL_PHASE:
            tensorboard_log_dir = osp.join(self.output_dir, epoch, prefix_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        if self.cfg.DEBUG.VIS_SKELETON:
            logger.info("=> VIS_SKELETON")
        if self.cfg.DEBUG.VIS_BBOX:
            logger.info("=> VIS_BBOX")
        return result_output_dir, vis_output_dir
