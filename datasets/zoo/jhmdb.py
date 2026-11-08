import random, cv2, torch, copy, warnings, json, logging
import os.path as osp
import numpy as np
from pycocotools.coco import COCO
from collections import OrderedDict

from datasets.process import half_body_transform, fliplr_joints, get_affine_transform, exec_affine_transform, generate_heatmaps
from datasets.zoo.base import VideoDataset
from datasets.transforms import build_transforms
from engine.defaults import TEST_PHASE, VAL_PHASE, TRAIN_PHASE
from utils.utils_bbox import box2cs
from datasets.zoo.jhmdb.cal_accuracy import keypoint_pck_accuracy
# from datasets import DATASET_REGISTRY
from utils.utils_registry import DATASET_REGISTRY
from utils.utils_image import read_image


@DATASET_REGISTRY.register()
class Jhmdb(VideoDataset):
    """
        Jhmdb
        0: "neck",
        1: "belly",
        2: "head",
        3: "right_shoulder",
        4: "left_shoulder",
        5: "right_hip",
        6: "left_hip",
        7: "right_elbow",
        8: "left_elbow",
        9: "right_knee",
        10: "left_knee",
        11: "right_wrist",
        12: "left_wrist",
        13: "right_ankle",
        14: "left_ankle"
    """

    def __init__(self, cfg, phase, **kwargs):
        super(Jhmdb, self).__init__(cfg, phase, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.train = True if phase == TRAIN_PHASE else False

        self.use_gt_bbox = True
        self.flip_pairs = [[3, 4], [5, 6], [7, 8], [9, 10], [9, 10], [11, 12], [13, 14]]
        self.upper_body_ids = (0, 1, 2, 3, 4, 7, 8, 11, 12)
        self.lower_body_ids = (5, 6, 9, 10, 13, 14)
        self.joint_weights = [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.2, 1.2, 1.5, 1.5, 1.5, 1.5]

        self.transform = build_transforms(cfg, phase)
        self.dataset_name = cfg.DATASET.NAME
        # self.sigmas = np.array([
        #     .25, 1.07, .25, .79, .79, 1.07, 1.07, .72, .72, .87, .87, .62, .62, .89, .89]) / 10.0
        self.sigma = cfg.MODEL.SIGMA

        self.distance_whole_otherwise_segment = True
        self.distance = cfg.DISTANCE

        self.random_aux_frame = cfg.DATASET.RANDOM_AUX_FRAME

        self.img_dir = cfg.DATASET.IMG_DIR
        self.img_prefix = self.img_dir
        self.bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        self.json_dir = cfg.DATASET.JSON_DIR

        Split_version = cfg.DATASET.SPLIT_VERSION
        if self.is_train:
            anno_file = osp.join(self.json_dir, f"Sub{Split_version}_train.json")
        else:
            anno_file = osp.join(self.json_dir, f"Sub{Split_version}_test.json")

        self.coco = COCO(anno_file)
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])
        self.image_set_index = self.coco.getImgIds()
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.num_images = len(self.image_set_index)

        self.data = self._get_db()

        self.model_input_type = cfg.DATASET.INPUT_TYPE
        self.logger.info(f"=> num_images : {self.num_images}")
        self.logger.info(f"=> load {len(self.data)} samples")

    def _get_db(self):
        """
            Load dataset.
        """
        assert self.use_gt_bbox
        db = self._load_coco_keypoint_annotations()
        return db

    def _load_coco_keypoint_annotations(self):
        """
                    coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
                    iscrowd:
                        crowd instances are handled by marking their overlaps with all categories to -1
                        and later excluded in training
                    bbox:
                        [x1, y1, w, h]
                """
        gt_db = []
        for img_id in self.image_set_index:
            im_ann = self.coco.loadImgs(img_id)[0]
            width = im_ann['width']
            height = im_ann['height']

            file_name = im_ann['file_name']

            nframes = int(im_ann['nframes'])
            frame_id = int(im_ann['frame_id'])

            annIds = self.coco.getAnnIds(imgIds=img_id)
            objs = self.coco.loadAnns(annIds)

            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                if 'bbox' not in obj:
                    continue
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if ('area' not in obj or obj['area'] > 0) and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            rec = []
            for obj in objs:
                if 'keypoints' not in obj:
                    continue
                cls = self._coco_ind_to_class_ind[obj['category_id']]
                if cls != 1:
                    continue

                # ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    continue

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
                })
            gt_db.extend(rec)
        return gt_db

    def evaluate(self, cfg, preds, res_folder, boxes, filenames, metric="PCK", **kwargs):
        # def evaluate(self, outputs, res_folder, metric="PCK", **kwargs):
        """Evaluate onehand10k keypoint results. The pose prediction results
        will be saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(preds, boxes, image_path, output_heatmap))
                :preds (np.ndarray[1,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[1,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_path (list[str])
                :output_heatmap (np.ndarray[N, K, H, W]): model outpus.

            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'PCK', 'tPCK'.
                PCK means normalized by the bounding boxes, while tPCK
                means normalized by the torso size.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'tPCK']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = osp.join(res_folder, 'result_keypoints.json')

        kpts = []

        for index in range(len(preds)):
            pred = preds[index]
            box = boxes[index]
            img_path = filenames[index]
            bbox_id = index

            image_id = self.name2id[img_path[len(self.img_prefix) + 1:]]
            kpts.append({
                'keypoints': pred.tolist(),
                'center': box[0:2].tolist(),
                'scale': box[2:4].tolist(),
                'area': box[4],
                'score': box[5],
                'image_id': image_id,
                'bbox_id': bbox_id
            })

        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value

    def _get_mapping_id_name(self, imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts

    def _write_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True)

    def _report_metric(self, res_file, metrics, pck_thr=0.2):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.data)

        outputs = []
        gts = []
        masks = []
        threshold_bbox = []
        threshold_torso = []

        for pred, item in zip(preds, self.data):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_vis'])[:, 0]) > 0)
            if 'PCK' in metrics:
                bbox = np.array(item['box'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))

            if 'tPCK' in metrics:
                torso_thr = np.linalg.norm(item['joints_3d'][4, :2] -
                                           item['joints_3d'][5, :2])
                if torso_thr < 1:
                    torso_thr = np.linalg.norm(
                        np.array(pred['keypoints'])[4, :2] -
                        np.array(pred['keypoints'])[5, :2])  #
                    warnings.warn('Torso Size < 1.')
                threshold_torso.append(np.array([torso_thr, torso_thr]))

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_torso = np.array(threshold_torso)

        if 'PCK' in metrics:
            pck_p, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr, threshold_bbox)

            stats_names = [
                'Head PCK', 'Sho PCK', 'Elb PCK', 'Wri PCK', 'Hip PCK',
                'Knee PCK', 'Ank PCK', 'Mean PCK'
            ]

            stats = [
                pck_p[2],
                0.5 * pck_p[3] + 0.5 * pck_p[4],
                0.5 * pck_p[7] + 0.5 * pck_p[8],
                0.5 * pck_p[11] + 0.5 * pck_p[12],
                0.5 * pck_p[5] + 0.5 * pck_p[6],
                0.5 * pck_p[9] + 0.5 * pck_p[10],
                0.5 * pck_p[13] + 0.5 * pck_p[14],
                pck
            ]

            info_str.extend(list(zip(stats_names, stats)))

        if 'tPCK' in metrics:
            pck_p, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                                  threshold_torso)

            stats_names = [
                'Head tPCK', 'Sho tPCK', 'Elb tPCK', 'Wri tPCK', 'Hip tPCK',
                'Knee tPCK', 'Ank tPCK', 'Mean tPCK'
            ]

            stats = [
                pck_p[2], 0.5 * pck_p[3] + 0.5 * pck_p[4],
                          0.5 * pck_p[7] + 0.5 * pck_p[8],
                          0.5 * pck_p[11] + 0.5 * pck_p[12],
                          0.5 * pck_p[5] + 0.5 * pck_p[6],
                          0.5 * pck_p[9] + 0.5 * pck_p[10],
                          0.5 * pck_p[13] + 0.5 * pck_p[14], pck
            ]

            info_str.extend(list(zip(stats_names, stats)))

        return info_str

    def __getitem__(self, item):
        data_item = copy.deepcopy(self.data[item])
        if self.model_input_type == 'single_frame':
            return self._get_single_frame(data_item)
        elif self.model_input_type == 'spatiotemporal_window':
            return self._get_spatiotemporal_window(data_item)

    def _get_single_frame(self, data_item):
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        data_numpy = read_image(image_file_path)

        if self.color_rgb:
            # cv2 read_image  color channel is BGR
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            self.logger.error('=> fail to read {}'.format(image_file_path))
            raise ValueError('Fail to read {}'.format(image_file_path))

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

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
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]

                joints, joints_vis = fliplr_joints(joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joint_weights)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)
        meta = {
            'image': image_file_path,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
        }

        return input_x, target_heatmaps, target_heatmaps_weight, meta

    def _get_spatiotemporal_window(self, data_item):
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        data_numpy = read_image(image_file_path)

        zero_fill = len(osp.basename(image_file_path).replace('.png', ''))
        current_idx = int(osp.basename(image_file_path).replace('.png', ''))

        # jhmdb frame_id start from one
        if self.distance_whole_otherwise_segment:
            # farthest_distance = 2
            farthest_distance = self.distance
            prev_delta_range = range(1, min((current_idx), farthest_distance))
            next_delta_range = range(1, min((num_frames - current_idx + 1), farthest_distance))
        else:
            # For example, if current_idx=0 , prev_delta = range(1,min(0+1,2))
            #  if current_idx=1, prev_delta= range(1,min(1+1,2))
            # 0,1,2,3,4   num_frames=5
            # if current_idx = 4 , 5 - 4 = 1.
            prev_delta_range = range(1, min(current_idx, self.previous_distance))
            next_delta_range = range(1, min((num_frames - current_idx + 1), self.next_distance))

        prev_delta_range = list(prev_delta_range)
        next_delta_range = list(next_delta_range)
        # setting deltas

        if len(prev_delta_range) == 0:
            prev_delta = 0
            margin_left = 1
        else:
            if self.random_aux_frame:
                prev_delta = random.choice(prev_delta_range)
            else:
                prev_delta = prev_delta_range[-1]
            margin_left = prev_delta

        if len(next_delta_range) == 0:
            next_delta = 0
            margin_right = 1
        else:
            if self.random_aux_frame:
                next_delta = random.choice(next_delta_range)
            else:
                next_delta = next_delta_range[-1]

            margin_right = next_delta

        prev_idx = current_idx - prev_delta
        next_idx = current_idx + next_delta

        prev_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx).zfill(zero_fill) + '.png')
        next_image_file = osp.join(osp.dirname(image_file_path), str(next_idx).zfill(zero_fill) + '.png')

        # checking for files existence
        if not osp.exists(prev_image_file):
            error_msg = "Can not find image :{}".format(prev_image_file)
            self.logger.error(error_msg)
            raise Exception(error_msg)
        if not osp.exists(next_image_file):
            error_msg = "Can not find image :{}".format(next_image_file)
            self.logger.error(error_msg)
            raise Exception(error_msg)

        data_numpy_prev = read_image(prev_image_file)
        data_numpy_next = read_image(next_image_file)

        if self.color_rgb:
            # cv2 read_image  color channel is BGR
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            data_numpy_prev = cv2.cvtColor(data_numpy_prev, cv2.COLOR_BGR2RGB)
            data_numpy_next = cv2.cvtColor(data_numpy_next, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            self.logger.error('=> fail to read {}'.format(image_file_path))
            raise ValueError('Fail to read {}'.format(image_file_path))
        if data_numpy_prev is None:
            self.logger.error('=> PREV SUP: fail to read {}'.format(prev_image_file))
            raise ValueError('PREV SUP: Fail to read {}'.format(prev_image_file))
        if data_numpy_next is None:
            self.logger.error('=> NEXT SUP: fail to read {}'.format(next_image_file))
            raise ValueError('NEXT SUP: Fail to read {}'.format(next_image_file))

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

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

                joints, joints_vis = fliplr_joints(joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_prev = cv2.warpAffine(data_numpy_prev, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_next = cv2.warpAffine(data_numpy_next, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)
            input_prev = self.transform(input_prev)
            input_next = self.transform(input_next)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joint_weights)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        meta = {
            'image': image_file_path,
            'prev_sup_image': prev_image_file,
            'next_sup_image': next_image_file,
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
        }

        return input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, meta
