#!/usr/bin/python
# -*- coding:utf8 -*-
"""
PoseTrack 2-frame tracking 数据集：仅使用「前一帧 + 当前帧」，用于 online 跟踪。
训练/验证时样本语义为 (prev, cur, cur)，即 next 复用 cur，与现有 3 路输入模型兼容。
"""

import numpy as np
import os.path as osp
import torch
import copy
import random
import cv2
from pycocotools.coco import COCO
import logging
from collections import OrderedDict
from tabulate import tabulate
from termcolor import colored

from .posetrack_utils import video2filenames, evaluate_simple
from utils.utils_json import read_json_from_file, write_json_to_file
from utils.utils_bbox import box2cs
from utils.utils_image import read_image
from utils.utils_folder import create_folder
from utils.utils_registry import DATASET_REGISTRY
from datasets.process import get_affine_transform, fliplr_joints, exec_affine_transform, generate_heatmaps, half_body_transform, \
    convert_data_to_annorect_struct

from datasets.transforms import build_transforms
from datasets.zoo.base import VideoDataset

from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE


@DATASET_REGISTRY.register()
class PoseTrack_2_frame_tracking(VideoDataset):
    """
    PoseTrack 2-frame tracking：只生成 prev + cur 两帧，next 复用 cur。
    输出仍为 3 路图像 (input_x, input_prev, input_next)，其中 input_next = input_x，
    便于与现有 3-frame 模型对齐，实现「只用前一帧+当前帧」的 online 跟踪。
    """

    def __init__(self, cfg, phase, **kwargs):
        super(PoseTrack_2_frame_tracking, self).__init__(cfg, phase, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.train = True if phase == TRAIN_PHASE else False
        self.flip_pairs = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.joints_weight = np.array([1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5],
                                      dtype=np.float32).reshape((self.num_joints, 1))
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.is_posetrack18 = cfg.DATASET.IS_POSETRACK18
        self.transform = build_transforms(cfg, phase)

        self.distance_whole_otherwise_segment = cfg.DISTANCE_WHOLE_OTHERWISE_SEGMENT
        self.distance = cfg.DISTANCE
        self.previous_distance = cfg.PREVIOUS_DISTANCE
        self.next_distance = cfg.NEXT_DISTANCE

        self.random_aux_frame = cfg.DATASET.RANDOM_AUX_FRAME

        self.bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        self.sigma = cfg.MODEL.SIGMA

        self.img_dir = cfg.DATASET.IMG_DIR
        self.json_dir = cfg.DATASET.JSON_DIR
        self.test_on_train = cfg.DATASET.TEST_ON_TRAIN
        self.json_file = cfg.DATASET.JSON_FILE

        if self.phase != TRAIN_PHASE:
            self.img_dir = cfg.DATASET.TEST_IMG_DIR
            temp_subCfgNode = cfg.VAL if self.phase == VAL_PHASE else cfg.TEST
            self.nms_thre = temp_subCfgNode.NMS_THRE
            self.image_thre = temp_subCfgNode.IMAGE_THRE
            self.soft_nms = temp_subCfgNode.SOFT_NMS
            self.oks_thre = temp_subCfgNode.OKS_THRE
            self.in_vis_thre = temp_subCfgNode.IN_VIS_THRE
            self.bbox_file = temp_subCfgNode.COCO_BBOX_FILE
            self.use_gt_bbox = temp_subCfgNode.USE_GT_BBOX
            self.annotation_dir = temp_subCfgNode.ANNOT_DIR

        self.coco = COCO(osp.join(self.json_dir, 'posetrack_train.json' if self.is_train else 'posetrack_val.json'))
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)

        self.data = self._list_data()

        self.model_input_type = cfg.DATASET.INPUT_TYPE

        self.show_data_parameters()
        self.show_samples()

    def __getitem__(self, item_index):
        data_item = copy.deepcopy(self.data[item_index])
        if self.model_input_type == 'single_frame':
            return self._get_single_frame(data_item)
        elif self.model_input_type == 'spatiotemporal_window':
            return self._get_spatiotemporal_window(data_item)

    def _get_spatiotemporal_window(self, data_item):
        """
        仅生成 prev + cur 两帧；next 复用 cur，使输出仍为 3 路 (prev, cur, cur)，与 3-frame 模型兼容。
        同时为前一帧生成 GT heatmap（用于后续 ST-MRF 的 loss 监督）。
        """
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        data_numpy = read_image(image_file_path)

        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))

        if zero_fill == 6:
            is_posetrack18 = True
        else:
            is_posetrack18 = False

        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))

        # 2-frame tracking：只使用前一帧，next 固定为当前帧
        if self.distance_whole_otherwise_segment:
            farthest_distance = self.distance
            prev_delta_range = range(1, min((current_idx + 1) if is_posetrack18 else current_idx, farthest_distance))
        else:
            prev_delta_range = range(1, min(current_idx + 1 if is_posetrack18 else current_idx, self.previous_distance))

        prev_delta_range = list(prev_delta_range)

        if len(prev_delta_range) == 0:
            prev_delta = 0
            margin_left = 1
        else:
            if self.random_aux_frame:
                prev_delta = random.choice(prev_delta_range)
            else:
                prev_delta = prev_delta_range[-1]
            margin_left = prev_delta

        # 固定 next = 当前帧，不读下一帧
        next_delta = 0
        margin_right = 1

        prev_idx = current_idx - prev_delta
        next_idx = current_idx  # next 复用当前帧

        prev_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx).zfill(zero_fill) + '.jpg')

        if not osp.exists(prev_image_file):
            error_msg = "Can not find image :{}".format(prev_image_file)
            self.logger.error(error_msg)
            raise Exception(error_msg)

        data_numpy_prev = read_image(prev_image_file)
        # next 复用当前帧，不额外读文件
        data_numpy_next = data_numpy.copy()
        
        # 从 data_item 中获取前一帧的 joints（已在数据加载阶段准备好）
        prev_joints_raw = data_item.get('prev_joints_3d', None)
        prev_joints_vis_raw = data_item.get('prev_joints_3d_vis', None)

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            data_numpy_prev = cv2.cvtColor(data_numpy_prev, cv2.COLOR_BGR2RGB)
            data_numpy_next = cv2.cvtColor(data_numpy_next, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            self.logger.error('=> fail to read {}'.format(image_file_path))
            raise ValueError('Fail to read {}'.format(image_file_path))
        if data_numpy_prev is None:
            self.logger.error('=> PREV SUP: fail to read {}'.format(prev_image_file))
            raise ValueError('PREV SUP: Fail to read {}'.format(prev_image_file))

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']
        
        # 前一帧的 joints（如果找到的话）
        prev_joints = prev_joints_raw.copy() if prev_joints_raw is not None else np.zeros((self.num_joints, 3), dtype=float)
        prev_joints_vis = prev_joints_vis_raw.copy() if prev_joints_vis_raw is not None else np.zeros((self.num_joints, 3), dtype=float)
        has_prev_gt = prev_joints_raw is not None

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints, self.upper_body_ids, self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                data_numpy_prev = data_numpy_prev[:, ::-1, :]
                data_numpy_next = data_numpy_next[:, ::-1, :]

                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                # 前一帧的 joints 也需要做同样的 flip
                if has_prev_gt:
                    prev_joints, prev_joints_vis = fliplr_joints(
                        prev_joints, prev_joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        trans = get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_prev = cv2.warpAffine(data_numpy_prev, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_next = cv2.warpAffine(data_numpy_next, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)
            input_prev = self.transform(input_prev)
            input_next = self.transform(input_next)

        # 对当前帧 joints 做仿射变换
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]

        # 对前一帧 joints 做同样的仿射变换（如果存在）
        if has_prev_gt:
            for i in range(self.num_joints):
                if prev_joints_vis[i, 0] > 0.0:
                    prev_joints[i, 0:2] = exec_affine_transform(prev_joints[i, 0:2], trans)

            for index, joint in enumerate(prev_joints):
                x, y, _ = joint
                if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                    prev_joints_vis[index] = [0, 0, 0]

        # 生成当前帧的 target heatmaps
        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)
        
        # 生成前一帧的 target heatmaps（用于 ST-MRF 的 loss 监督）
        # 借鉴 PoseTrack.py：prev_heatmaps 是 numpy array，不需要单独的 weight
        if has_prev_gt:
            prev_heatmaps, _ = generate_heatmaps(
                prev_joints, prev_joints_vis, self.sigma, self.image_size, self.heatmap_size,
                self.num_joints)
            # 注意：保持为 numpy array，与 PoseTrack.py 一致（虽然注释掉了 torch.from_numpy，但保持格式一致）
        else:
            # 如果前一帧没有 GT，生成全零的 heatmaps
            prev_heatmaps = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

        meta = {
            'image': image_file_path,
            'prev_sup_image': prev_image_file,
            'next_sup_image': image_file_path,  # next 复用当前帧
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
            'margin_left': margin_left,
            'margin_right': margin_right,
            'labeled_prev': 1 if has_prev_gt else 0,  # 借鉴 PoseTrack.py 的命名
        }

        # 返回格式与 PoseTrack.py 一致：
        # input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, prev_heatmaps, next_heatmaps, meta
        # 注意：prev_heatmaps 是 numpy array（与 PoseTrack.py 一致），next_heatmaps 复用 prev_heatmaps（因为 next=cur）
        return input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, \
               prev_heatmaps, prev_heatmaps, meta  # next_heatmaps 复用 prev_heatmaps（因为 next=cur）

    def _get_single_frame(self, data_item):
        raise NotImplementedError

    def _list_data(self):
        if self.is_train or self.use_gt_bbox:
            data, video_db = self._load_coco_keypoints_annotations()
            # 训练阶段：过滤掉每个视频的第一个标注帧（因为需要前一帧的target做loss）
            if self.is_train:
                data = self._filter_first_annotated_frame(data)
            # 为每个样本添加前一帧的 joints（如果存在）
            data = self._add_prev_frame_joints(data, video_db)
        else:
            data = self._load_detection_results()
        return data

    def _filter_first_annotated_frame(self, data):
        """
        训练阶段过滤掉每个视频的第一个标注帧。
        因为 ST-MRF 需要前一帧的 target heatmap 做 loss 监督，而第一个标注帧没有前一帧 GT。
        
        Args:
            data: 原始数据列表，每个元素包含 'image', 'frame_id' 等字段
        
        Returns:
            过滤后的数据列表（已移除所有视频的第一个标注帧）
        """
        if len(data) == 0:
            return data
        
        # 按视频分组：从 image 路径提取视频名（目录路径）
        video_to_samples = {}
        for item in data:
            # image 路径格式：./Dataset/PoseTrack2017/images_renamed/video_name/xxxxx.jpg
            # 或：path/to/video_name/xxxxx.jpg
            image_path = item['image']
            video_name = osp.dirname(image_path)  # 获取目录路径作为视频名
            if video_name not in video_to_samples:
                video_to_samples[video_name] = []
            video_to_samples[video_name].append(item)
        
        # 对每个视频，找出最小的 frame_id，过滤掉所有 frame_id == min_frame_id 的样本
        filtered_data = []
        num_filtered = 0
        for video_name, samples in video_to_samples.items():
            if len(samples) == 0:
                continue
            # 找出该视频的最小 frame_id
            min_frame_id = min(s['frame_id'] for s in samples)
            # 只保留 frame_id > min_frame_id 的样本
            for sample in samples:
                if sample['frame_id'] > min_frame_id:
                    filtered_data.append(sample)
                else:
                    num_filtered += 1
        
        self.logger.info(
            f"=> Training phase: Filtered out {num_filtered} samples (first annotated frame per video). "
            f"Remaining samples: {len(filtered_data)} (originally {len(data)})"
        )
        
        return filtered_data

    def _add_prev_frame_joints(self, data, video_db):
        """
        为每个样本添加前一帧的 joints（如果存在）。
        借鉴 PoseTrack.py 的方式，在数据加载阶段就准备好 prev_joints。

        注意：
        - video_db 里存的 db_ids 是「原始 gt_db 列表」中的下标
        - data 在 _filter_first_annotated_frame 之后已经被过滤、重新排列
        - 因此需要先建立 original_db_id -> 当前 data 下标 的映射，再用 video_db 查找

        Args:
            data: 过滤后的数据列表（每个样本包含 'db_id'，表示其在原始 gt_db 中的索引）
            video_db: 视频数据库，格式为 video_db[video_name][person_id] = {'frame_ids': [...], 'db_ids': [...]}

        Returns:
            添加了 prev_joints_3d 和 prev_joints_3d_vis 的数据列表
        """
        # 先建立「原始 db_id -> 当前 data 下标」的映射
        orig2new_index = {}
        for new_idx, item in enumerate(data):
            db_id = item.get('db_id', None)
            if db_id is not None:
                orig2new_index[db_id] = new_idx

        for item_idx, item in enumerate(data):
            # 从 image 路径提取 video_name（与 _load_coco_keypoints_annotations 中的方式一致）
            file_name = item['image'].replace(self.img_dir + '/', '').replace(self.img_dir + '\\', '')
            video_name = osp.basename(osp.dirname(file_name))  # 与 PoseTrack.py 一致
            frame_id = item['frame_id']

            # 尝试从 video_db 中找到同一人的前一帧
            prev_joints, prev_joints_vis = None, None
            if video_name in video_db:
                # 遍历该视频的所有 person_id，找到包含当前 frame_id 的人
                for person_id, info in video_db[video_name].items():
                    frame_ids = info['frame_ids']
                    db_ids = info['db_ids']
                    if frame_id in frame_ids:
                        idx = frame_ids.index(frame_id)
                        # 查找前一帧（同一人、frame_id 更小），并且该帧在过滤后仍然存在
                        for i in range(idx - 1, -1, -1):
                            if frame_ids[i] < frame_id:
                                orig_db_id = db_ids[i]
                                if orig_db_id in orig2new_index:
                                    prev_item = data[orig2new_index[orig_db_id]]
                                    prev_joints = prev_item['joints_3d'].copy()
                                    prev_joints_vis = prev_item['joints_3d_vis'].copy()
                                    break
                        break

            item['prev_joints_3d'] = prev_joints
            item['prev_joints_3d_vis'] = prev_joints_vis

        return data

    def _load_coco_keypoints_annotations(self):
        """
        加载 COCO 标注，返回数据列表和 video_db（用于匹配同一人的不同帧）。
        借鉴 PoseTrack.py 的实现方式。
        """
        gt_db = []
        video_db = {}
        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']

            file_name = im_ann['file_name']

            nframes = int(im_ann['nframes'])
            frame_id = int(im_ann['frame_id'])

            annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = self.coco.loadAnns(annIds)

            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            rec = []
            for obj in objs:
                cls = self._coco_ind_to_class_ind[obj['category_id']]
                if cls != 1:
                    continue

                if max(obj['keypoints']) == 0:
                    continue

                joints_3d = np.zeros((self.num_joints, 3), dtype=float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=float)
                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                center, scale = box2cs(obj['clean_bbox'][:4], self.aspect_ratio, self.bbox_enlarge_factor)

                # 构建 video_db（用于匹配同一人的不同帧），与 PoseTrack.py 一致
                video_name = osp.basename(osp.dirname(file_name))  # 与 PoseTrack.py 一致
                person_id = obj.get('track_id', 0)  # 如果没有 track_id，用 0

                # 这里的 db_id 是在「原始 gt_db 列表」里的下标
                db_id = len(gt_db) + len(rec)
                if video_name not in video_db:
                    video_db[video_name] = {}
                if person_id not in video_db[video_name]:
                    video_db[video_name][person_id] = {'frame_ids': [], 'db_ids': []}
                video_db[video_name][person_id]['frame_ids'].append(frame_id)
                video_db[video_name][person_id]['db_ids'].append(db_id)

                # 同时把 db_id 写进样本本身，方便在「过滤之后」重新建立映射
                rec.append({
                    'image': osp.join(self.img_dir, file_name),
                    'center': center,
                    'scale': scale,
                    'box': obj['clean_bbox'][:4],
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    'nframes': nframes,
                    'frame_id': frame_id,
                    'db_id': db_id,
                })
            gt_db.extend(rec)
        return gt_db, video_db

    def _load_detection_results(self):
        logger = logging.getLogger(__name__)
        logger.info("=> Load bbox file from {}".format(self.bbox_file))
        all_boxes = read_json_from_file(self.bbox_file)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        kpt_data = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = det_res['image_name']
            box = det_res['bbox']
            score = det_res['score']
            nframes = det_res['nframes']
            frame_id = det_res['frame_id']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = box2cs(box, self.aspect_ratio, self.bbox_enlarge_factor)
            joints_3d = np.zeros((self.num_joints, 3), dtype=float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=float)
            kpt_data.append({
                'image': osp.join(self.img_dir, img_name),
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'nframes': nframes,
                'frame_id': frame_id,
            })

        table_header = ["Total boxes", "Filter threshold", "Remaining boxes"]
        table_data = [[len(all_boxes), self.image_thre, num_boxes]]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Boxes Info Table: \n" + colored(table, "magenta"))

        return kpt_data

    def evaluate(self, cfg, preds, output_dir, boxes, img_path, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info("=> Start evaluate")
        if self.phase == 'validate':
            output_dir = osp.join(output_dir, 'val_set_json_results')
        else:
            output_dir = osp.join(output_dir, 'test_set_json_results')

        create_folder(output_dir)

        video_map = {}
        vid2frame_map = {}
        vid2name_map = {}

        all_preds = []
        all_boxes = []
        cc = 0

        for key in img_path:
            temp = key.split('/')

            video_name = temp[len(temp) - 3] + '/' + temp[len(temp) - 2]
            img_sfx = temp[len(temp) - 3] + '/' + temp[len(temp) - 2] + '/' + temp[len(temp) - 1]

            prev_nm = temp[len(temp) - 1]
            frame_num = int(prev_nm.replace('.jpg', ''))
            if not video_name in video_map:
                video_map[video_name] = [cc]
                vid2frame_map[video_name] = [frame_num]
                vid2name_map[video_name] = [img_sfx]
            else:
                video_map[video_name].append(cc)
                vid2frame_map[video_name].append(frame_num)
                vid2name_map[video_name].append(img_sfx)

            idx_list = img_path[key]
            pose_list = []
            box_list = []
            for idx in idx_list:
                temp = np.zeros((4, 17))
                temp[0, :] = preds[idx, :, 0]
                temp[1, :] = preds[idx, :, 1]
                temp[2, :] = preds[idx, :, 2]
                temp[3, :] = preds[idx, :, 2]
                pose_list.append(temp)

                temp = np.zeros((1, 6))
                temp[0, :] = boxes[idx, :]
                box_list.append(temp)

            all_preds.append(pose_list)
            all_boxes.append(box_list)
            cc += 1

        annot_dir = self.annotation_dir
        is_posetrack18 = self.is_posetrack18

        out_data = {}
        out_filenames, L = video2filenames(annot_dir)

        for vid in video_map:
            idx_list = video_map[vid]
            c = 0
            used_frame_list = []
            cur_length = L['images/' + vid]

            temp_kps_map = {}
            temp_box_map = {}

            for idx in idx_list:
                frame_num = vid2frame_map[vid][c]
                img_sfx = vid2name_map[vid][c]
                c += 1

                used_frame_list.append(frame_num)

                kps = all_preds[idx]
                temp_kps_map[frame_num] = (img_sfx, kps)

                bb = all_boxes[idx]
                temp_box_map[frame_num] = bb

            if not is_posetrack18:
                sid = 1
                fid = cur_length + 1
            else:
                sid = 0
                fid = cur_length

            for current_frame_id in range(sid, fid):
                frame_num = current_frame_id
                if not current_frame_id in used_frame_list:
                    temp_sfx = vid2name_map[vid][0]
                    arr = temp_sfx.split('/')
                    if not is_posetrack18:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(8) + '.jpg'
                    else:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(6) + '.jpg'
                    kps = []
                    tracks = []
                    bboxs = []

                else:

                    img_sfx = temp_kps_map[frame_num][0]
                    kps = temp_kps_map[frame_num][1]
                    bboxs = temp_box_map[frame_num]
                    tracks = [track_id for track_id in range(len(kps))]

                data_el = {
                    'image': {'name': img_sfx},
                    'imgnum': [frame_num],
                    'annorect': convert_data_to_annorect_struct(kps, tracks, bboxs),
                }
                if vid in out_data:
                    out_data[vid].append(data_el)
                else:
                    out_data[vid] = [data_el]

        logger.info("=> saving files for evaluation")
        for vname in out_data:
            vdata = out_data[vname]
            outfpath = osp.join(output_dir, out_filenames[osp.join('images', vname)])

            write_json_to_file({'annolist': vdata}, outfpath)

        AP = evaluate_simple.evaluate(annot_dir, output_dir, eval_track=False)[0]

        name_value = [
            ('Head', AP[0]),
            ('Shoulder', AP[1]),
            ('Elbow', AP[2]),
            ('Wrist', AP[3]),
            ('Hip', AP[4]),
            ('Knee', AP[5]),
            ('Ankle', AP[6]),
            ('Mean', AP[7])
        ]

        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

    def evaluate_tracking(self, cfg, preds, output_dir, boxes, img_path, track_ids, *args, **kwargs):
        """
        跟踪评估方法：接收带真实 track_id 的预测，写入 JSON 并调用 eval_track=True。
        
        Args:
            cfg: 配置对象
            preds: 预测结果，shape 为 (N, num_joints, 3)
            output_dir: 输出目录
            boxes: 边界框，shape 为 (N, 6)
            img_path: 字典，key 为图像路径，value 为该图像中预测的索引列表
            track_ids: 字典，key 为图像路径（与 img_path 的 key 一致），value 为该图像中每个预测对应的真实 track_id 列表
            *args, **kwargs: 其他参数
        
        Returns:
            name_value: 评估指标字典
            mean_value: 平均指标值
        """
        logger = logging.getLogger(__name__)
        logger.info("=> Start evaluate tracking")
        if self.phase == 'validate':
            output_dir = osp.join(output_dir, 'val_set_json_results')
        else:
            output_dir = osp.join(output_dir, 'test_set_json_results')

        create_folder(output_dir)

        video_map = {}
        vid2frame_map = {}
        vid2name_map = {}

        all_preds = []
        all_boxes = []
        all_track_ids = []  # 存储每个预测对应的 track_id
        cc = 0

        for key in img_path:
            temp = key.split('/')

            video_name = temp[len(temp) - 3] + '/' + temp[len(temp) - 2]
            img_sfx = temp[len(temp) - 3] + '/' + temp[len(temp) - 2] + '/' + temp[len(temp) - 1]

            prev_nm = temp[len(temp) - 1]
            frame_num = int(prev_nm.replace('.jpg', ''))
            if not video_name in video_map:
                video_map[video_name] = [cc]
                vid2frame_map[video_name] = [frame_num]
                vid2name_map[video_name] = [img_sfx]
            else:
                video_map[video_name].append(cc)
                vid2frame_map[video_name].append(frame_num)
                vid2name_map[video_name].append(img_sfx)

            idx_list = img_path[key]
            pose_list = []
            box_list = []
            track_id_list = []
            
            # 获取该图像对应的 track_id 列表
            if key not in track_ids:
                logger.warning(f"=> No track_ids provided for image {key}, using sequential IDs")
                frame_track_ids = [track_id for track_id in range(len(idx_list))]
            else:
                frame_track_ids = track_ids[key]
                if len(frame_track_ids) != len(idx_list):
                    logger.warning(f"=> track_ids length ({len(frame_track_ids)}) != idx_list length ({len(idx_list)}) for {key}, using sequential IDs")
                    frame_track_ids = [track_id for track_id in range(len(idx_list))]
            
            for idx_idx, idx in enumerate(idx_list):
                temp = np.zeros((4, 17))
                temp[0, :] = preds[idx, :, 0]
                temp[1, :] = preds[idx, :, 1]
                temp[2, :] = preds[idx, :, 2]
                temp[3, :] = preds[idx, :, 2]
                pose_list.append(temp)

                temp = np.zeros((1, 6))
                temp[0, :] = boxes[idx, :]
                box_list.append(temp)
                
                # 存储对应的 track_id
                track_id_list.append(frame_track_ids[idx_idx])

            all_preds.append(pose_list)
            all_boxes.append(box_list)
            all_track_ids.append(track_id_list)
            cc += 1

        annot_dir = self.annotation_dir
        is_posetrack18 = self.is_posetrack18

        out_data = {}
        out_filenames, L = video2filenames(annot_dir)

        for vid in video_map:
            idx_list = video_map[vid]
            c = 0
            used_frame_list = []
            cur_length = L['images/' + vid]

            temp_kps_map = {}
            temp_box_map = {}
            temp_track_ids_map = {}  # 存储每帧的 track_id

            for idx in idx_list:
                frame_num = vid2frame_map[vid][c]
                img_sfx = vid2name_map[vid][c]
                c += 1

                used_frame_list.append(frame_num)

                kps = all_preds[idx]
                temp_kps_map[frame_num] = (img_sfx, kps)

                bb = all_boxes[idx]
                temp_box_map[frame_num] = bb
                
                # 存储该帧的 track_id
                temp_track_ids_map[frame_num] = all_track_ids[idx]

            if not is_posetrack18:
                sid = 1
                fid = cur_length + 1
            else:
                sid = 0
                fid = cur_length

            for current_frame_id in range(sid, fid):
                frame_num = current_frame_id
                if not current_frame_id in used_frame_list:
                    temp_sfx = vid2name_map[vid][0]
                    arr = temp_sfx.split('/')
                    if not is_posetrack18:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(8) + '.jpg'
                    else:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(6) + '.jpg'
                    kps = []
                    tracks = []
                    bboxs = []

                else:
                    img_sfx = temp_kps_map[frame_num][0]
                    kps = temp_kps_map[frame_num][1]
                    bboxs = temp_box_map[frame_num]
                    # 使用真实的 track_id
                    tracks = temp_track_ids_map[frame_num]

                data_el = {
                    'image': {'name': img_sfx},
                    'imgnum': [frame_num],
                    'annorect': convert_data_to_annorect_struct(kps, tracks, bboxs),
                }
                if vid in out_data:
                    out_data[vid].append(data_el)
                else:
                    out_data[vid] = [data_el]

        logger.info("=> saving files for tracking evaluation")
        for vname in out_data:
            vdata = out_data[vname]
            outfpath = osp.join(output_dir, out_filenames[osp.join('images', vname)])

            write_json_to_file({'annolist': vdata}, outfpath)

        # 调用跟踪评估，eval_track=True
        AP, track_metrics = evaluate_simple.evaluate(annot_dir, output_dir, eval_pose=True, eval_track=True)

        # 姿态估计指标（AP 是一个列表，长度为 8：0-6 是各部位，7 是平均）
        name_value = [
            ('Head', AP[0] if len(AP) > 0 else np.nan),
            ('Shoulder', AP[1] if len(AP) > 1 else np.nan),
            ('Elbow', AP[2] if len(AP) > 2 else np.nan),
            ('Wrist', AP[3] if len(AP) > 3 else np.nan),
            ('Hip', AP[4] if len(AP) > 4 else np.nan),
            ('Knee', AP[5] if len(AP) > 5 else np.nan),
            ('Ankle', AP[6] if len(AP) > 6 else np.nan),
            ('Mean', AP[7] if len(AP) > 7 else np.nan)
        ]

        # 跟踪指标（track_metrics 也是一个列表，由 getCum 返回）
        # 索引 0-6 是各部位的 MOTA，7 是平均 MOTA，8 是 MOTP，9 是 Precision，10 是 Recall
        if track_metrics is not None and len(track_metrics) > 0:
            name_value.extend([
                ('MOTA', track_metrics[7] if len(track_metrics) > 7 else np.nan),
                ('MOTP', track_metrics[8] if len(track_metrics) > 8 else np.nan),
                ('Precision', track_metrics[9] if len(track_metrics) > 9 else np.nan),
                ('Recall', track_metrics[10] if len(track_metrics) > 10 else np.nan),
            ])

        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
