#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/26
    Description:
        弱监督中的对比实验，
        （需要用到模型）
        直接使用最邻近的姿态作为当前帧的姿态

"""
import copy
import logging
import os.path as osp
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from datasets.process import get_affine_transform, fliplr_joints, exec_affine_transform, generate_heatmaps, half_body_transform, \
    convert_data_to_annorect_struct
from datasets.transforms import build_transforms
from datasets.zoo.base import VideoDataset
from datasets.zoo.posetrack.posetrack_utils import video2filenames, evaluate_simple
from engine.defaults.constant import DATASET_REGISTRY
from utils.common import TRAIN_PHASE, VAL_PHASE
from utils.utils_bbox import box2cs, cs2box
from utils.utils_folder import create_folder
from utils.utils_image import read_image
from utils.utils_json import write_json_to_file


@DATASET_REGISTRY.register()
class PoseTrack_WS_Five_Frames_Interpolation_new_7frame(VideoDataset):
    """
        PoseTrack_WS_Five_Frames_Interpolation_new_7frame
    """

    def __init__(self, cfg, phase, **kwargs):
        super(PoseTrack_WS_Five_Frames_Interpolation_new_7frame, self).__init__(cfg, phase, **kwargs)
        self.logger = logging.getLogger(__name__)

        self.phase = phase

        ## 只能是验证阶段
        # assert self.phase == VAL_PHASE

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

        self.coco = COCO(osp.join(self.json_dir, 'posetrack_train.json' if self.is_train else 'posetrack_val_consecutive.json'))
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)

        self.data, self.video_db, self.video_labeled_frame_ids = self._list_data()
        self.T = cfg.DATASET.TIME_INTERVAL
        if phase == TRAIN_PHASE:
            self.data = self._train_generate_samples(self.data, self.video_db, self.T)
        if phase == VAL_PHASE:
            # 在验证集上测试时，每隔几帧筛选, 把不需要标注
            self.logger.info("T {}".format(str(self.T)))
            self.data, self.sup_frame_db = self._val_generate_samples(self.data, self.video_db, self.video_labeled_frame_ids, self.T)
            self.logger.info("num of data {} ; num of sup frame {}".format(len(self.data), len(self.sup_frame_db)))
            self.sup_images_info = self.prepare_for_eval(self.sup_frame_db)

        self.model_input_type = cfg.DATASET.INPUT_TYPE

        self.show_data_parameters()
        self.show_samples()

    def prepare_for_eval(self, sup_frame_db):
        # 不按视频，直接按图片分
        sup_images_info = dict()
        for sup_frame_item in sup_frame_db:
            image = sup_frame_item['image']
            temp = image.split('/')
            image_name = '{}/{}/{}'.format(temp[-3], temp[-2], temp[-1])
            # box = sup_frame_item['box']
            center = sup_frame_item['center']
            scale = sup_frame_item['scale']
            area = scale[0] * 200 * scale[1] * 200
            box = [center[0], center[1], scale[0], scale[1], area, 1.0]
            pose = sup_frame_item['joints_3d']
            pose_vis = sup_frame_item['joints_3d_vis']
            pose[:, 2] = pose_vis[:, 0]

            if image_name not in sup_images_info.keys():
                sup_images_info[image_name] = {"box": [], "pose": []}

            sup_images_info[image_name]['box'].append(box)
            sup_images_info[image_name]['pose'].append(pose)

        return sup_images_info

    def _train_generate_samples(self, data, video_db, T):
        generated_samples = []

        debug = False

        for item in data:

            # 训练的时候不应该用当前帧的信息
            video_name, cur_frame_id, person_id = item['video_name'], item['frame_id'], item['person_id']
            instance_labeled_info = video_db[video_name][person_id]
            frame_ids = instance_labeled_info['frame_ids']
            db_ids = instance_labeled_info['db_ids']

            prev_frame_id, next_frame_id = cur_frame_id, cur_frame_id
            candidates_prev_frame_ids = []
            candidates_next_frame_ids = []
            for frame_id in frame_ids:
                # 先前帧为临近 当前帧最近的
                if frame_id < prev_frame_id and (frame_id + (T - 1)) >= cur_frame_id:
                    candidates_prev_frame_ids.append(frame_id)
                    # prev_frame_id = frame_id
                if frame_id > next_frame_id and (frame_id - (T - 1)) <= cur_frame_id:
                    candidates_next_frame_ids.append(frame_id)
                    # next_frame_id = frame_id

            if len(candidates_prev_frame_ids) > 0:
                prev_frame_id = random.choice(candidates_prev_frame_ids)#随机选一帧

            if len(candidates_next_frame_ids) > 0:
                next_frame_id = (prev_frame_id + T) if (prev_frame_id + T) in candidates_next_frame_ids else random.choice(
                    candidates_next_frame_ids)

            if prev_frame_id == cur_frame_id and next_frame_id == cur_frame_id:
                prev_frame_id = max(1, cur_frame_id - 1)
                next_frame_id = min(item['nframes'], cur_frame_id + 1)

                prev_image = item['image'].replace(str(cur_frame_id).zfill(8), str(prev_frame_id).zfill(8))
                next_image = item['image'].replace(str(cur_frame_id).zfill(8), str(next_frame_id).zfill(8))
                union_center, union_scale = item['center'], item['scale']

                prev_joints, prev_joints_vis = None, None
                next_joints, next_joints_vis = None, None
                interpolation_joints, interpolation_joints_vis = None, None
            elif prev_frame_id != cur_frame_id and next_frame_id == cur_frame_id:
                # 只有前帧
                prev_frame_id = random.choice(candidates_prev_frame_ids)
                prev_db_item = data[db_ids[frame_ids.index(prev_frame_id)]]
                union_center, union_scale = prev_db_item['center'], prev_db_item['scale']
                next_frame_id = min(item['nframes'], cur_frame_id + 1)

                prev_image = prev_db_item['image']
                next_image = item['image'].replace(str(cur_frame_id).zfill(8), str(next_frame_id).zfill(8))

                prev_joints, prev_joints_vis = prev_db_item['joints_3d'], prev_db_item['joints_3d_vis']
                next_joints, next_joints_vis = None, None

                interpolation_joints, interpolation_joints_vis = None, None
            elif prev_frame_id == cur_frame_id and next_frame_id != cur_frame_id:
                # 只有后帧
                next_frame_id = random.choice(candidates_next_frame_ids)
                next_db_item = data[db_ids[frame_ids.index(next_frame_id)]]
                union_center, union_scale = next_db_item['center'], next_db_item['scale']
                prev_frame_id = max(1, cur_frame_id - 1)

                prev_image = item['image'].replace(str(cur_frame_id).zfill(8), str(prev_frame_id).zfill(8))
                next_image = next_db_item['image']

                prev_joints, prev_joints_vis = None, None
                next_joints, next_joints_vis = next_db_item['joints_3d'], next_db_item['joints_3d_vis']
                interpolation_joints, interpolation_joints_vis = None, None

            else:
                # 能做插值的
                prev_db_item = data[db_ids[frame_ids.index(prev_frame_id)]]
                next_db_item = data[db_ids[frame_ids.index(next_frame_id)]]

                prev_image, next_image = prev_db_item['image'], next_db_item['image']

                prev_center, prev_scale = prev_db_item['center'], prev_db_item['scale']
                next_center, next_scale = next_db_item['center'], next_db_item['scale']

                prev_box = cs2box(prev_center, prev_scale, pattern='xyxy')
                next_box = cs2box(next_center, next_scale, pattern='xyxy')

                union_box_xyxy = [min(prev_box[0], next_box[0]), min(prev_box[1], next_box[1]),
                                  max(prev_box[2], next_box[2]), max(prev_box[3], next_box[3])]
                x, y, x2, y2 = union_box_xyxy
                union_box_xywh = [x, y, x2 - x, y2 - y]
                union_center, union_scale = box2cs(union_box_xywh, self.aspect_ratio)
                prev_distance = cur_frame_id - prev_frame_id
                prev_joints, prev_joints_vis = prev_db_item['joints_3d'], prev_db_item['joints_3d_vis']
                next_joints, next_joints_vis = next_db_item['joints_3d'], next_db_item['joints_3d_vis']
                interpolation_joints = np.zeros((self.num_joints, 3), dtype=np.float64)
                interpolation_joints_vis = np.zeros((self.num_joints, 3), dtype=np.float64)
                for joint_id in range(self.num_joints):
                    prev_joint, prev_joint_vis = prev_joints[joint_id], prev_joints_vis[joint_id][0]
                    next_joint, next_joint_vis = next_joints[joint_id], next_joints_vis[joint_id][0]
                    if prev_joint_vis == 1 and next_joint_vis == 0:
                        interpolation_joints_vis[joint_id] = np.array([1, 1, 0])
                        interpolation_joints[joint_id] = prev_joint
                    elif prev_joint_vis == 0 and next_joint_vis == 1:
                        interpolation_joints_vis[joint_id] = np.array([1, 1, 0])
                        interpolation_joints[joint_id] = next_joint
                    elif prev_joint_vis == 1 and next_joint_vis == 1:
                        interpolation_joints_vis[joint_id] = np.array([1, 1, 0])
                        interpolation_joints[joint_id] = prev_joint + (
                                ((next_joint - prev_joint) * prev_distance) / (next_frame_id - prev_frame_id))

            generated_samples.append({
                'cur_image': item['image'],
                'nframes': item['nframes'],
                'prev_image': prev_image,
                'next_image': next_image,
                'center': union_center,
                'scale': union_scale,
                'prev_joints_3d': prev_joints,
                'prev_joints_3d_vis': prev_joints_vis,
                'next_joints_3d': next_joints,
                'next_joints_3d_vis': next_joints_vis,
                'joints_3d': item['joints_3d'],
                'joints_3d_vis': item['joints_3d_vis'],
                'frame_id': item['frame_id'],
                'margin_left': max(1, cur_frame_id - prev_frame_id),
                'margin_right': max(1, next_frame_id - cur_frame_id),
                'person_id': person_id,
                'video_name': video_name,
                'interpolation_joints_3d': interpolation_joints,
                'interpolation_joints_3d_vis': interpolation_joints_vis
            })

        return generated_samples

    def _val_generate_samples(self, data, video_db, video_labeled_frame_ids, T):

        ## TODO
        ## 需要修改
        """
            尽量确保样本个数的不变的
        :param data:
        :param video_db:
        :param T:
        :return:
        """
        from collections import defaultdict

        video_sup_frame_idx = defaultdict(list)
        # " 59, 60, 61, 62, ...., 87,88,89"
        # => 59, 64, 69, 74, 79, 84, 89
        for video_name, labeled_frame_idx in video_labeled_frame_ids.items():
            # sid = labeled_frame_idx[2]
            sid = labeled_frame_idx[0]
            # sid = labeled_frame_idx[0]
            fid = labeled_frame_idx[-1]
            sup_ids = []
            while sid <= fid:
                sup_ids.append(sid)
                sid = sid + T

            # if fid not in sup_ids:
            #     sup_ids.append(fid)
            video_sup_frame_idx[video_name] = sup_ids

        sup_frame_db = []
        generated_samples = []
        counter = [0, 0, 0, 0]
        interpolation_counter = [0, 0, 0, 0]
        sup_frame_counter = [0, 0]
        import copy

        for item_index, item in enumerate(data):
            video_name, cur_frame_id, person_id = item['video_name'], item['frame_id'], item['person_id']
            instance_labeled_info = video_db[video_name][person_id]
            frame_ids = instance_labeled_info['frame_ids']
            db_ids = instance_labeled_info['db_ids']

            sup_frame_ids = video_sup_frame_idx[video_name]

            if cur_frame_id in sup_frame_ids:
                # 如果当前帧是辅助帧就保存在辅助帧samples
                sup_frame_db.append(copy.deepcopy(item))
                sup_frame_counter[0] += 1
            else:
                # 非辅助帧的姿态，用最邻近的辅助帧来替换。
                # 因为此方法不用使用模型，就可以直接用最邻近的辅助帧来替换
                sup_frame_counter[1] += 1
                prev_frame_id, next_frame_id = cur_frame_id, cur_frame_id
                for frame_id in sup_frame_ids:
                    # 先前帧为临近 当前帧最近的
                    if frame_id < prev_frame_id and (frame_id + (T - 1)) >= cur_frame_id and frame_id in frame_ids:
                        prev_frame_id = frame_id
                    if frame_id > next_frame_id and (frame_id - (T - 1)) <= cur_frame_id and frame_id in frame_ids:
                        next_frame_id = frame_id

                temp_item = copy.deepcopy(item)

                if prev_frame_id != cur_frame_id and next_frame_id != cur_frame_id:
                    # 如果前后帧都能用 GT 标注
                    prev_db_item = data[db_ids[frame_ids.index(prev_frame_id)]]
                    next_db_item = data[db_ids[frame_ids.index(next_frame_id)]]

                    prev_image, next_image = prev_db_item['image'], next_db_item['image']

                    prev_center, prev_scale = prev_db_item['center'], prev_db_item['scale']
                    next_center, next_scale = next_db_item['center'], next_db_item['scale']

                    prev_box = cs2box(prev_center, prev_scale, pattern='xyxy')
                    next_box = cs2box(next_center, next_scale, pattern='xyxy')

                    union_box_xyxy = [min(prev_box[0], next_box[0]), min(prev_box[1], next_box[1]),
                                      max(prev_box[2], next_box[2]), max(prev_box[3], next_box[3])]
                    x, y, x2, y2 = union_box_xyxy
                    union_box_xywh = [x, y, x2 - x, y2 - y]
                    union_center, union_scale = box2cs(union_box_xywh, self.aspect_ratio)

                    prev_distance = cur_frame_id - prev_frame_id

                    prev_joints, prev_joints_vis = prev_db_item['joints_3d'], prev_db_item['joints_3d_vis']
                    next_joints, next_joints_vis = next_db_item['joints_3d'], next_db_item['joints_3d_vis']

                    cur_joints = np.zeros((self.num_joints, 3), dtype=np.float64)
                    cur_joints_vis = np.zeros((self.num_joints, 3), dtype=np.float64)
                    for joint_id in range(self.num_joints):
                        prev_joint, prev_joint_vis = prev_joints[joint_id], prev_joints_vis[joint_id][0]
                        next_joint, next_joint_vis = next_joints[joint_id], next_joints_vis[joint_id][0]
                        if prev_joint_vis == 1 and next_joint_vis == 0:
                            cur_joints_vis[joint_id] = np.array([1, 1, 0])
                            cur_joints[joint_id] = prev_joint
                            interpolation_counter[0] += 1
                        elif prev_joint_vis == 0 and next_joint_vis == 1:
                            cur_joints_vis[joint_id] = np.array([1, 1, 0])
                            cur_joints[joint_id] = next_joint
                            interpolation_counter[1] += 1
                        elif prev_joint_vis == 1 and next_joint_vis == 1:
                            cur_joints_vis[joint_id] = np.array([1, 1, 0])
                            cur_joints[joint_id] = prev_joint + (
                                    ((next_joint - prev_joint) * prev_distance) / (next_frame_id - prev_frame_id))
                            interpolation_counter[2] += 1
                        else:
                            interpolation_counter[3] += 1

                    interpolation_joints = cur_joints
                    interpolation_joints_vis = cur_joints_vis
                    counter[0] += 1
                elif prev_frame_id == cur_frame_id and next_frame_id != cur_frame_id:
                    # 当前帧为t
                    # 如果只有后帧有标注
                    # 那么前帧使用t-1,后帧使用标注
                    next_db_item = data[db_ids[frame_ids.index(next_frame_id)]]

                    union_center, union_scale = next_db_item['center'], next_db_item['scale']
                    prev_frame_id = max(1, cur_frame_id - 1)

                    prev_image = item['image'].replace(str(cur_frame_id).zfill(8), str(prev_frame_id).zfill(8))
                    next_image = next_db_item['image']

                    prev_joints, prev_joints_vis = None, None
                    next_joints, next_joints_vis = next_db_item['joints_3d'], next_db_item['joints_3d_vis']
                    interpolation_joints, interpolation_joints_vis = None, None
                elif prev_frame_id != cur_frame_id and prev_frame_id == cur_frame_id:
                    # 当前帧为t
                    # 标注只有前帧
                    # 前帧使用标注帧，后帧使用t+1帧
                    prev_db_item = data[db_ids[frame_ids.index(prev_frame_id)]]
                    union_center, union_scale = prev_db_item['center'], prev_db_item['scale']
                    next_frame_id = min(item['nframes'], cur_frame_id + 1)

                    # 图片
                    prev_image = prev_db_item['image']
                    next_image = item['image'].replace(str(cur_frame_id).zfill(8), str(next_frame_id).zfill(8))
                    # 关键点
                    prev_joints, prev_joints_vis = prev_db_item['joints_3d'], prev_db_item['joints_3d_vis']
                    next_joints, next_joints_vis = None, None
                    interpolation_joints, interpolation_joints_vis = None, None

                    counter[2] += 1
                else:
                    #
                    prev_frame_id = max(1, cur_frame_id - 1)
                    next_frame_id = min(item['nframes'], cur_frame_id + 1)

                    prev_image = item['image'].replace(str(cur_frame_id).zfill(8), str(prev_frame_id).zfill(8))
                    next_image = item['image'].replace(str(cur_frame_id).zfill(8), str(next_frame_id).zfill(8))
                    union_center, union_scale = item['center'], item['scale']

                    prev_joints, prev_joints_vis = None, None
                    next_joints, next_joints_vis = None, None
                    interpolation_joints, interpolation_joints_vis = None, None

                    counter[3] += 1

                generated_samples.append({
                    'cur_image': item['image'],
                    'nframes': item['nframes'],
                    'prev_image': prev_image,
                    'next_image': next_image,
                    'center': union_center,
                    'scale': union_scale,
                    'prev_joints_3d': prev_joints,
                    'prev_joints_3d_vis': prev_joints_vis,
                    'next_joints_3d': next_joints,
                    'next_joints_3d_vis': next_joints_vis,
                    'joints_3d': item['joints_3d'],  # 没用的
                    'joints_3d_vis': item['joints_3d_vis'],  # 没用的
                    'frame_id': item['frame_id'],
                    'margin_left': max(1, cur_frame_id - prev_frame_id),
                    'margin_right': max(1, next_frame_id - cur_frame_id),
                    'person_id': person_id,
                    'video_name': video_name,
                    'interpolation_joints_3d': interpolation_joints,
                    'interpolation_joints_3d_vis': interpolation_joints_vis
                })
        self.logger.info("Num sup frame {} ; Num Interpolation frame {}".format(sup_frame_counter[0], sup_frame_counter[1]))
        self.logger.info(
            "Perfect {} ;Miss Left {}; Miss Right {}; Miss Left and Right {}".format(counter[0], counter[1], counter[2], counter[3]))
        self.logger.info(
            "Interpolation {} {} {} {} ; {} {} {} {}".format(interpolation_counter[0], interpolation_counter[1],
                                                             interpolation_counter[2], interpolation_counter[3],
                                                             interpolation_counter[0] / 17, interpolation_counter[1] / 17,
                                                             interpolation_counter[2] / 17, interpolation_counter[3] / 17)
        )
        return generated_samples, sup_frame_db

    def __getitem__(self, item_index):
        data_item = copy.deepcopy(self.data[item_index])
        if self.model_input_type == 'single_frame':
            return self._get_single_frame(data_item)
        elif self.model_input_type == 'spatiotemporal_window':

            ###
            input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight, prev_heatmaps, next_heatmaps, interpolation_heatmaps, \
            prev_heatmaps_list,next_heatmaps_list, meta = self._get_spatiotemporal_window(
                data_item)

            # input_x, output_order, output_reverse, target_heatmaps, target_heatmaps_weight, meta = self._get_spatiotemporal_window(
            #     data_item)
            prev_heatmaps_list = torch.cat(prev_heatmaps_list, dim=0)
            next_heatmaps_list = torch.cat(next_heatmaps_list, dim=0)  # todo    output_reverser?

            # return input_x, prev_neighbor_ref_x, next_neighbor_ref_x, target_heatmaps, target_heatmaps_weight, meta
            return input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight, prev_heatmaps, next_heatmaps, interpolation_heatmaps, \
                   prev_heatmaps_list, next_heatmaps_list, meta
            ###

            return self._get_spatiotemporal_window(data_item)

    def _get_spatiotemporal_window(self, data_item):##之前是取前兩幀和后兩幀，現在要改為前7幀后7幀
        # Data for pose refinement
        cur_image_path, prev_image_path, next_image_path = data_item['cur_image'], data_item['prev_image'], data_item['next_image']
        center, scale = data_item['center'], data_item['scale']
        joints, joints_vis = data_item['joints_3d'], data_item['joints_3d_vis']
        prev_joints, prev_joints_vis = data_item['prev_joints_3d'], data_item['prev_joints_3d_vis']
        next_joints, next_joints_vis = data_item['next_joints_3d'], data_item['next_joints_3d_vis']
        interpolation_joints, interpolation_joints_vis = data_item['interpolation_joints_3d'], data_item['interpolation_joints_3d_vis']
        labeled_prev, labeled_next = 1, 1
        interpolation = 1
        if prev_joints is None:
            prev_joints = np.zeros((17, 3))
            prev_joints_vis = np.zeros((17, 3))
            labeled_prev = 0

        if next_joints is None:
            next_joints = np.zeros((17, 3))
            next_joints_vis = np.zeros((17, 3))
            labeled_next = 0

        if interpolation_joints is None:
            interpolation_joints = np.zeros((17, 3))
            interpolation_joints_vis = np.zeros((17, 3))
            interpolation = 0

        # cur_frame_id = data_item['frame_id']
        num_frames = data_item['nframes']
        zero_fill = len(osp.basename(cur_image_path).replace('.jpg', ''))

        if zero_fill == 6:
            is_posetrack18 = True
        else:
            is_posetrack18 = False

        # Data for temporal infer
        current_idx = int(osp.basename(cur_image_path).replace('.jpg', ''))

        # farthest_distance = 3
        farthest_distance = self.distance

        # left_delta_range = list(range(1, min(current_idx - 1, farthest_distance)))
        # right_delta_range = list(range(1, min(num_frames - current_idx + 1, farthest_distance)))
        # 取两帧就行

        prev_delta_range = list(range(1, min((current_idx + 1) if is_posetrack18 else current_idx, farthest_distance)))
        next_delta_range = list(range(1,
                                 min((num_frames - current_idx) if is_posetrack18 else (num_frames - current_idx + 1),
                                     farthest_distance)))

        # prev_delta_1, prev_delta_2 = 0, 0  # prev_delta_2 比 prev_delta_1 远
        # next_delta_1, next_delta_2 = 0, 0  # next_delta_2 比 next_delta_1 远
        prev_delta_sup_range, next_delta_sup_range = [], []
        if len(prev_delta_range) == 0:
            # 0个 previous frame，补n个当前帧
            for i in range(self.distance - 1 - len(prev_delta_range)):
                prev_delta_sup_range.append(0)
        else:
            # 有x个previous frame,但不够n个，补n-x个最临近的previous frame
            for i in range(self.distance - 1 - len(prev_delta_range)):
                prev_delta_sup_range.append(1)

        if len(next_delta_range) == 0:

            for i in range(self.distance - 1 - len(next_delta_range)):
                next_delta_sup_range.append(0)
        else:
            for i in range(self.distance - 1 - len(next_delta_range)):
                next_delta_sup_range.append(1)

        # if len(left_delta_range) == 1:
        #     prev_delta_1 = left_delta_range[0]
        #     prev_delta_2 = left_delta_range[0]
        # elif len(left_delta_range) == 2:
        #     prev_delta_1 = left_delta_range[0]
        #     prev_delta_2 = left_delta_range[1]
        #
        # if len(right_delta_range) == 1:
        #     next_delta_1 = right_delta_range[0]
        #     next_delta_2 = right_delta_range[1]
        # elif len(right_delta_range) == 2:
        #     next_delta_1 = right_delta_range[0]
        #     next_delta_2 = right_delta_range[1]

        # Suppose current_idx = 20
        # prev_image_file1 = osp.join(osp.dirname(cur_image_path), str(current_idx - prev_delta_1).zfill(zero_fill) + '.jpg')  # 19
        # prev_image_file2 = osp.join(osp.dirname(cur_image_path), str(current_idx - prev_delta_2).zfill(zero_fill) + '.jpg')  # 18
        # next_image_file1 = osp.join(osp.dirname(cur_image_path), str(current_idx + next_delta_1).zfill(zero_fill) + '.jpg')  # 21
        # next_image_file2 = osp.join(osp.dirname(cur_image_path), str(current_idx + next_delta_2).zfill(zero_fill) + '.jpg')  # 22
        #
        # prev_image1_data_numpy = read_image(prev_image_file1)
        # prev_image2_data_numpy = read_image(prev_image_file2)
        # next_image1_data_numpy = read_image(next_image_file1)
        # next_image2_data_numpy = read_image(next_image_file2)
        #
        cur_data_numpy = read_image(cur_image_path)
        prev_data_numpy = read_image(prev_image_path)
        next_data_numpy = read_image(next_image_path)
        #
        # if not osp.exists(prev_image_path) or not osp.exists(cur_image_path) or not osp.exists(next_image_path):
        #     error_msg = "Can not find prev_image:{}, cur_image:{}, next_image:{}".format(prev_image_path, cur_image_path,
        #                                                                                  next_image_path)
        #
        #     self.logger.error(error_msg)
        #     raise Exception(error_msg)

        prev_delta_range.reverse()
        prev_delta_range.extend(prev_delta_sup_range)
        next_delta_sup_range.extend(next_delta_range)
        next_delta_range = next_delta_sup_range

        ids = OrderedDict()
        ids["prev_supp_idx"] = []
        ids["next_supp_idx"] = []
        ids["current_idx"] = current_idx

        prev_ref_image_files = []
        for prev_delta in prev_delta_range:
            prev_input_id = current_idx - prev_delta
            prev_image_file = osp.join(osp.dirname(cur_image_path), str(prev_input_id).zfill(zero_fill) + '.jpg')
            prev_ref_image_files.append(prev_image_file)
            ids["prev_supp_idx"].append(prev_input_id)

        next_ref_image_files = []
        for next_delta in next_delta_range:
            next_input_id = current_idx + next_delta
            next_image_file = osp.join(osp.dirname(cur_image_path),
                                       str(next_input_id).zfill(zero_fill) + '.jpg')
            next_ref_image_files.append(next_image_file)
            ids["next_supp_idx"].append(next_input_id)

        prev_ref_data_numpy_list = []
        next_ref_data_numpy_list = []

        for prev_ref_image_file in prev_ref_image_files:
            if not osp.exists(prev_ref_image_file):
                error_msg = "Can not find image :{}".format(prev_ref_image_file)
                self.logger.error(error_msg)
                raise Exception(error_msg)
            prev_ref_data_numpy = read_image(prev_ref_image_file)
            if self.color_rgb:
                # cv2 read_image  color channel is BGR
                prev_ref_data_numpy = cv2.cvtColor(prev_ref_data_numpy, cv2.COLOR_BGR2RGB)
            if prev_ref_data_numpy is None:
                self.logger.error('=> fail to read {}'.format(prev_ref_image_file))
                raise ValueError('Fail to read {}'.format(prev_ref_image_file))
            prev_ref_data_numpy_list.append(prev_ref_data_numpy)

        for next_ref_image_file in next_ref_image_files:
            if not osp.exists(next_ref_image_file):
                error_msg = "Can not find image :{}".format(next_ref_image_file)
                self.logger.error(error_msg)
                raise Exception(error_msg)
            ref_data_numpy = read_image(next_ref_image_file)
            if self.color_rgb:
                # cv2 read_image  color channel is BGR
                ref_data_numpy = cv2.cvtColor(ref_data_numpy, cv2.COLOR_BGR2RGB)
            if ref_data_numpy is None:
                self.logger.error('=> fail to read {}'.format(next_ref_image_file))
                raise ValueError('Fail to read {}'.format(next_ref_image_file))
            next_ref_data_numpy_list.append(ref_data_numpy)
        if self.color_rgb:
            cur_data_numpy = cv2.cvtColor(cur_data_numpy, cv2.COLOR_BGR2RGB)#转换成灰度图
            prev_data_numpy = cv2.cvtColor(prev_data_numpy, cv2.COLOR_BGR2RGB)
            next_data_numpy = cv2.cvtColor(next_data_numpy, cv2.COLOR_BGR2RGB)

            # prev_image1_data_numpy = cv2.cvtColor(prev_image1_data_numpy, cv2.COLOR_BGR2RGB)
            # prev_image2_data_numpy = cv2.cvtColor(prev_image2_data_numpy, cv2.COLOR_BGR2RGB)
            # next_image1_data_numpy = cv2.cvtColor(next_image1_data_numpy, cv2.COLOR_BGR2RGB)
            # next_image2_data_numpy = cv2.cvtColor(next_image2_data_numpy, cv2.COLOR_BGR2RGB)

        if cur_data_numpy is None:
            self.logger.error('=> fail to read {}'.format(cur_image_path))
            raise ValueError('Fail to read {}'.format(cur_image_path))
        if prev_data_numpy is None:
            self.logger.error('=> PREV SUP: fail to read {}'.format(prev_image_path))
            raise ValueError('PREV SUP: Fail to read {}'.format(prev_image_path))
        if next_data_numpy is None:
            self.logger.error('=> NEXT SUP: fail to read {}'.format(next_image_path))
            raise ValueError('NEXT SUP: Fail to read {}'.format(next_image_path))
        for prev_image1_data_numpy in prev_ref_data_numpy_list:
            if prev_image1_data_numpy is None:
                self.logger.error('=> Prev Image 1: fail to read {}'.format(prev_image_file))
                raise ValueError('Prev Image 1: Fail to read {}'.format(next_image_path))
            # if prev_image2_data_numpy is None:
            #     self.logger.error('=> Prev Image 2: fail to read {}'.format(prev_image_file2))
            #     raise ValueError('Prev Image 2: Fail to read {}'.format(next_image_path))
        for next_image1_data_numpy in next_ref_data_numpy_list:
            if next_image1_data_numpy is None:
                self.logger.error('=> Next Image 1: fail to read {}'.format(next_image_file))
                raise ValueError('Next Image 1: Fail to read {}'.format(next_image_path))
            # if next_image2_data_numpy is None:
            #     self.logger.error('=> Next Image 1: fail to read {}'.format(next_image_file2))
            #     raise ValueError('Next Image 2: Fail to read {}'.format(next_image_path))

        r = 0
        if self.is_train:
            if np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.randn() < self.prob_half_body:
                half_body_center, half_body_scale = half_body_transform(joints, joints_vis,
                                                                        self.num_joints, self.upper_body_ids,
                                                                        self.aspect_ratio, self.pixel_std)
                if half_body_center is None or half_body_scale is None:
                    pass
                else:
                    center, scale = half_body_center, half_body_scale

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor

            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            random_scale_factor = np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            scale = scale * random_scale_factor

            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                cur_data_numpy = cur_data_numpy[:, ::-1, :]
                prev_data_numpy = prev_data_numpy[:, ::-1, :]
                next_data_numpy = next_data_numpy[:, ::-1, :]
                # prev_image1_data_numpy = prev_image1_data_numpy[:, ::-1, :]
                # prev_image2_data_numpy = prev_image2_data_numpy[:, ::-1, :]
                # next_image1_data_numpy = next_image1_data_numpy[:, ::-1, :]
                # next_image2_data_numpy = next_image2_data_numpy[:, ::-1, :]

                for i in range(len(prev_ref_data_numpy_list)):
                    prev_ref_data_numpy_list[i] = prev_ref_data_numpy_list[i][:, ::-1, :]
                for i in range(len(next_ref_data_numpy_list)):
                    next_ref_data_numpy_list[i] = next_ref_data_numpy_list[i][:, ::-1, :]

                image_width = cur_data_numpy.shape[1]
                joints, joints_vis = fliplr_joints(joints, joints_vis, image_width, self.flip_pairs)
                prev_joints, prev_joints_vis = fliplr_joints(prev_joints, prev_joints_vis, image_width, self.flip_pairs)
                next_joints, next_joints_vis = fliplr_joints(next_joints, next_joints_vis, image_width, self.flip_pairs)

                center[0] = cur_data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        cur_data_numpy = cv2.warpAffine(cur_data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        prev_data_numpy = cv2.warpAffine(prev_data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        next_data_numpy = cv2.warpAffine(next_data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        # prev_image1_data_numpy = cv2.warpAffine(prev_image1_data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        # prev_image2_data_numpy = cv2.warpAffine(prev_image2_data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        # next_image1_data_numpy = cv2.warpAffine(next_image1_data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        # next_image2_data_numpy = cv2.warpAffine(next_image2_data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        for i in range(len(prev_ref_data_numpy_list)):###
            prev_ref_data_numpy_list[i] = cv2.warpAffine(prev_ref_data_numpy_list[i], trans,
                                                    (int(self.image_size[0]), int(self.image_size[1])),
                                                    flags=cv2.INTER_LINEAR)
        for i in range(len(next_ref_data_numpy_list)):
            next_ref_data_numpy_list[i] = cv2.warpAffine(next_ref_data_numpy_list[i], trans,
                                                    (int(self.image_size[0]), int(self.image_size[1])),
                                                    flags=cv2.INTER_LINEAR)

        input_x, input_prev, input_next = cur_data_numpy, prev_data_numpy, next_data_numpy
        # Suppose current frame  t
        # input_forward_infer_frame1 = prev_image1_data_numpy  # t-1
        # input_forward_infer_frame2 = prev_image2_data_numpy  # t-2
        # input_backward_infer_frame1 = next_image1_data_numpy  # t+1
        # input_backward_infer_frame2 = next_image2_data_numpy  # t+2
        if self.transform:
            input_x = self.transform(cur_data_numpy)
            input_prev = self.transform(prev_data_numpy)
            input_next = self.transform(next_data_numpy)
            # input_forward_infer_frame1 = self.transform(input_forward_infer_frame1)
            # input_forward_infer_frame2 = self.transform(input_forward_infer_frame2)
            # input_backward_infer_frame1 = self.transform(input_backward_infer_frame1)
            # input_backward_infer_frame2 = self.transform(input_backward_infer_frame2)
            for i in range(len(prev_ref_data_numpy_list)):
                prev_ref_data_numpy_list[i] = self.transform(prev_ref_data_numpy_list[i])

            for i in range(len(next_ref_data_numpy_list)):
                next_ref_data_numpy_list[i] = self.transform(next_ref_data_numpy_list[i])

        # if self.is_train:
        # joint transform like image
        # 关键点从全局到局部
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)
            if prev_joints_vis[i, 0] > 0.0:
                prev_joints[i, 0:2] = exec_affine_transform(prev_joints[i, 0:2], trans)
            if next_joints_vis[i, 0] > 0.0:
                next_joints[i, 0:2] = exec_affine_transform(next_joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]

        for index, joint in enumerate(prev_joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                prev_joints_vis[index] = [0, 0, 0]

        for index, joint in enumerate(next_joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                next_joints_vis[index] = [0, 0, 0]

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        prev_heatmaps, _ = generate_heatmaps(prev_joints, prev_joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                             self.num_joints)
        next_heatmaps, _ = generate_heatmaps(next_joints, next_joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                             self.num_joints)
        interpolation_heatmaps, _ = generate_heatmaps(interpolation_joints, interpolation_joints_vis, self.sigma, self.image_size,
                                                      self.heatmap_size,
                                                      self.num_joints)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)
        prev_heatmaps = torch.from_numpy(prev_heatmaps)
        next_heatmaps = torch.from_numpy(next_heatmaps)

        meta = {
            'image': cur_image_path,
            'prev_sup_image': prev_image_path,
            'next_sup_image': next_image_path,
            'joints': joints,
            'joints_vis': joints_vis,
            'labeled_prev': labeled_prev,
            'labeled_next': labeled_next,
            'center': center,
            'scale': scale,
            'rotation': r,
            'margin_left': data_item['margin_left'],
            'margin_right': data_item['margin_right'],
            'score': 1.0,
            'interpolation': interpolation,

        }
        return input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, prev_heatmaps, next_heatmaps, interpolation_heatmaps, \
               prev_ref_data_numpy_list, next_ref_data_numpy_list, meta
               # input_forward_infer_frame1, input_forward_infer_frame2, input_backward_infer_frame1, input_backward_infer_frame2, meta

    def _get_single_frame(self, data_item):
        raise NotImplementedError

    def _list_data(self):
        assert self.is_train or self.use_gt_bbox
        return self._load_coco_keypoints_annotations()

    def _load_coco_keypoints_annotations(self):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
        """
        gt_db = []
        video_db = {}
        video_labeled_frame_ids = {}
        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']

            file_name = im_ann['file_name']

            video_name = osp.basename(osp.dirname(file_name))
            nframes = int(im_ann['nframes'])
            frame_id = int(im_ann['frame_id'])  # frame_id = file_name

            annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = self.coco.loadAnns(annIds)

            gt_db_start_id = len(gt_db)
            rec = []
            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']  # x y wh
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    # 有效的bbox
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]  # x y w h

                    # ignore non-person class
                    cls = self._coco_ind_to_class_ind[obj['category_id']]
                    if cls != 1:
                        continue

                    # ignore objs without keypoints annotation
                    if max(obj['keypoints']) == 0:
                        continue

                    db_id = gt_db_start_id + len(rec)  # start from zero

                    person_id = obj['track_id']
                    video_table = video_db.get(video_name, dict())
                    video_person_table = video_table.get(person_id, dict())
                    video_person_frame_ids = video_person_table.get("frame_ids", list())
                    video_person_db_ids = video_person_table.get("db_ids", list())

                    video_person_frame_ids.append(frame_id)
                    video_person_db_ids.append(db_id)
                    video_person_table["frame_ids"] = video_person_frame_ids
                    video_person_table["db_ids"] = video_person_db_ids
                    video_table[person_id] = video_person_table
                    video_db[video_name] = video_table

                    joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                    joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
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
                    #
                    center, scale = box2cs(obj['clean_bbox'][:4], self.aspect_ratio, self.bbox_enlarge_factor)

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
                        'person_id': person_id,
                        'video_name': video_name
                    })

            if len(rec) > 0:
                #
                labeled_frame_ids = video_labeled_frame_ids.get(video_name, list())
                if frame_id not in labeled_frame_ids:
                    labeled_frame_ids.append(frame_id)
                    video_labeled_frame_ids[video_name] = labeled_frame_ids

            gt_db.extend(rec)
        return gt_db, video_db, video_labeled_frame_ids

    def evaluate(self, cfg, preds, output_dir, boxes, img_path, *args, **kwargs):
        """

        :param cfg:
        :param preds:
        :param output_dir:
        :param boxes:
        :param img_path: 是字典 。  key 为 图片名称， value为对应的序号
        :param args:
        :param kwargs:
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.info("=> Start evaluate")
        if self.phase == 'validate':
            output_dir = osp.join(output_dir, 'val_set_json_results')
        else:
            output_dir = osp.join(output_dir, 'test_set_json_results')

        create_folder(output_dir)

        ### processing our preds
        video_map = {}
        vid2frame_map = {}
        vid2name_map = {}

        all_preds = []
        all_boxes = []
        cc = 0

        for key in img_path:  #
            temp = key.split('/')

            # video_name = osp.dirname(key)
            video_name = temp[len(temp) - 3] + '/' + temp[len(temp) - 2]
            img_sfx = temp[len(temp) - 3] + '/' + temp[len(temp) - 2] + '/' + temp[len(temp) - 1]

            prev_nm = temp[len(temp) - 1]
            frame_num = int(prev_nm.replace('.jpg', ''))
            if video_name not in video_map:
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

        # 添加辅助帧
        for image_key in self.sup_images_info:
            # e.g. image_key 'mpii_5sec/014523_mpii/00000017.jpg'
            temp = image_key.split('/')  # ['mpii_5sec', '014523_mpii', '00000017.jpg']
            video_name = '{}/{}'.format(temp[0], temp[1])  # 'mpii_5sec/014523_mpii'
            img_sfx = image_key  # 'mpii_5sec/014523_mpii/00000017.jpg'

            prev_nm = temp[-1]  # '00000017.jpg'
            frame_num = int(prev_nm.replace('.jpg', ''))  # 17

            if video_name not in video_map:
                video_map[video_name] = [cc]
                vid2frame_map[video_name] = [frame_num]
                vid2name_map[video_name] = [img_sfx]
            else:
                video_map[video_name].append(cc)
                vid2frame_map[video_name].append(frame_num)
                vid2name_map[video_name].append(img_sfx)

            temp_box_list = self.sup_images_info[image_key]['box']
            temp_pose_list = self.sup_images_info[image_key]['pose']

            assert len(temp_box_list) == len(temp_pose_list)

            pose_list = []
            box_list = []
            for i in range(len(temp_box_list)):
                temp = np.zeros((4, 17))
                pose = temp_pose_list[i]
                temp[0, :] = pose[:, 0]
                temp[1, :] = pose[:, 1]
                temp[2, :] = pose[:, 2]
                temp[3, :] = pose[:, 2]
                pose_list.append(temp)

                temp = np.zeros((1, 6))
                box = temp_box_list[i]
                temp[0, :] = box
                box_list.append(temp)
            all_preds.append(pose_list)
            all_boxes.append(box_list)
            cc += 1

        annot_dir = self.annotation_dir
        is_posetrack18 = self.is_posetrack18

        out_data = {}
        # 两个字典
        # out_filenames : {video : out_file_name}
        # L             : {video : num_frame}
        out_filenames, L = video2filenames(annot_dir)

        for vid in video_map:
            idx_list = video_map[vid]
            c = 0
            used_frame_list = []
            cur_length = L['images/' + vid]

            temp_kps_map = {}
            temp_track_kps_map = {}
            temp_box_map = {}

            for idx in idx_list:
                frame_num = vid2frame_map[vid][c]  # 编号
                img_sfx = vid2name_map[vid][c]
                c += 1

                used_frame_list.append(frame_num)

                kps = all_preds[idx]
                temp_kps_map[frame_num] = (img_sfx, kps)

                bb = all_boxes[idx]
                temp_box_map[frame_num] = bb
            #### including empty frames
            nnz_counter = 0
            next_track_id = 0

            if not is_posetrack18:
                sid = 1
                fid = cur_length + 1
            else:
                sid = 0
                fid = cur_length
            # start id ~ finish id
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
                    # tracks = [1] * len(kps)

                ### creating a data element
                data_el = {
                    'image': {'name': img_sfx},
                    'imgnum': [frame_num],
                    'annorect': convert_data_to_annorect_struct(kps, tracks, bboxs),
                }
                if vid in out_data:
                    out_data[vid].append(data_el)
                else:
                    out_data[vid] = [data_el]

        ##

        logger.info("=> saving files for evaluation")
        #### saving files for evaluation
        for vname in out_data:
            vdata = out_data[vname]
            outfpath = osp.join(output_dir, out_filenames[osp.join('images', vname)])

            write_json_to_file({'annolist': vdata}, outfpath)

        # run evaluation
        # AP = self._run_eval(annot_dir, output_dir)[0]
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
