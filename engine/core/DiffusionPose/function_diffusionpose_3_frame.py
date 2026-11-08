#!/usr/bin/python
# -*- coding:utf8 -*-
import time
import torch
import numpy as np
import os.path as osp
import logging
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
from ..evaludate import accuracy, pck_accuracy, pck_accuracy_origin_image
from engine.core import CORE_FUNCTION_REGISTRY, BaseFunction, AverageMeter
import torch
import torch.nn.functional as F

# from engine.evaludate import accuracy
from engine.defaults import VAL_PHASE, TEST_PHASE, TRAIN_PHASE
from datasets.process import get_final_preds
from datasets.transforms import reverse_transforms
from datasets.process import get_final_preds, dark_get_final_preds
from utils.utils_bbox import cs2box
from utils.utils_folder import create_folder
from utils.utils_image_tensor import tensor2im


from torch.nn.functional import kl_div, softmax
from tabulate import tabulate

def cnt_grad(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm().item()

    return total_norm


def _test(model):
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c , h , w = 1, 3 * 3, 256 , 192
    random_x = torch.randn((b, c , h  , w )).to('cuda')

    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params:,}")
    print(f"Model FLOPS #: {profile_macs(model, random_x):,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(random_x)

    import time
    num_iterations = 100
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time

    print(f"FPS: {fps}")

@CORE_FUNCTION_REGISTRY.register()
class Function_DiffusionPose_3_frame(BaseFunction):

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

        self.PE_Name = kwargs.get("PE_Name", "DiffusionPOSE")
        ##
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0
        self.softmax = torch.nn.Softmax(dim=1)
        self.DataSetName = str(self.cfg.DATASET.NAME).upper()
        self.mlp = torch.nn.Linear(cfg.MODEL.VIT.EMB_DIM , cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0]).cuda()

    def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, **kwargs):
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
        # switch to train mode
        model.train()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        end = time.time()



        for iter_step in range(self.max_iter_num):
            input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight ,  meta = next(self.dataloader_iter)

            self._before_train_iter(input_x)

            data_time.update(time.time() - end)

            margin_left = meta["margin_left"]
            margin_right = meta["margin_right"]
            margin = torch.stack([margin_left, margin_right], dim=1).cuda()
            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)




            concat_input = torch.cat((input_x, input_sup_A, input_sup_B), 1).cuda()
            outputs , htms_backbone ,feat_curr , fused_feature ,  kpt_tokens_curr = model(concat_input , target = target_heatmaps, margin = margin )




            if isinstance(outputs, list) or isinstance(outputs, tuple):
                pred_heatmaps = outputs[0]
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
                for pred_heatmaps in outputs[1:]:
                    loss += self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
            else:
                pred_heatmaps = outputs
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)

            ##mi_loss
            mi_loss_1 = self.feat_label_mi_estimation_V1(target_heatmaps, fused_feature, model) \
                        - self.feat_feat_mi_estimation(fused_feature, feat_curr)
            mi_loss_2 = self.feat_label_mi_estimation_keypoint_gt(target_heatmaps , kpt_tokens_curr)
            # mi_loss_2 = self.feat_label_mi_estimation_V2(target_heatmaps, htms_merg_mask_cat, model) \
            #             - self.feat_label_mi_estimation_V2(htms_curr, htms_merg_mask_cat, model)
            alpha = 1
            beta = 0.1
            gamma = 0.001
            # mi_feat = (-1) * beta * mi_loss_1 + (-1) * beta * alpha * mi_loss_2
            mi_feat = (-1) * beta * mi_loss_1 + (-1) * gamma * mi_loss_2
            # mi_feat = (-1) * beta * mi_loss_1
            mi_loss = alpha * mi_feat
            loss += mi_loss

            edge_loss = self.edge_potential_loss(pred_heatmaps, target_heatmaps,  lambda_length=1.0,  lambda_angle=1.0 )
            edge_loss= 0.0001 * edge_loss
            loss +=  edge_loss

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            grad.update(cnt_grad(model))
            optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), input_x.size(0))
            MI_FEAT.update(mi_feat.item(), input_x.size(0))
            EDGE_LOSS.update(edge_loss.item(), input_x.size(0))


            _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
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
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(epoch, iter_step, self.max_iter_num, batch_time=batch_time,
                                                                        speed=input_x.size(0) / batch_time.val,
                                                                        data_time=data_time, grad = grad,loss=losses, acc=acc , MI_FEAT = MI_FEAT,EDGE_LOSS = EDGE_LOSS)

                logger.info(msg)

            # For Tensorboard
            self.tb_writer.add_scalar('train_loss', losses.val, self.global_steps)
            self.tb_writer.add_scalar('train_acc', acc.val, self.global_steps)
            self.global_steps += 1

        tb_writer_dict["global_steps"] = self.global_steps

        # mi_loss


    def decode_heatmap_to_coords(self, heatmaps):
        """
        从热图中解码关节点的坐标。
        Args:
            heatmaps: 形状为 (B, J, 64, 48) 的热图
        Returns:
            coords: 形状为 (B, J, 2) 的关节点坐标，2 表示 (x, y) 坐标
        """
        B, J, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, J, -1)
        idx = torch.argmax(heatmaps, dim=-1)
        y_coords = (idx // W + 1) / H * 1.00
        x_coords = (idx % W + 1) / W  * 1.00
        coords = torch.stack([x_coords, y_coords], dim=-1)

        return coords


    def edge_potential_loss(self, pred_heatmaps, target_heatmaps,   lambda_length=1.0,  lambda_angle=1.0 ):
        """
        计算基于边势函数的损失，约束肢体长度和关节角度。

        Args:
            pred_heatmaps: 预测的热图, 形状为 (B, J, 64, 48)
            target_heatmaps: 真实热图, 形状为 (B, J, 64, 48)
            target_limb_lengths: 目标肢体长度, 形状为 (B, num_limbs)
            target_angles: 目标肢体角度, 形状为 (B, num_angles)

        Returns:
            edge_loss: 马尔科夫边势损失
        """
        B, J, H, W = pred_heatmaps.shape
        limb_pairs  = [(0, 1),(0, 2),(1, 2),(1, 3),(2, 4),(3, 5),(4, 6),(5, 7),(5, 11),(6, 8),(6, 12),(7, 9),(8, 10),(11, 13),(12, 14),(13,15),(14, 16)]
        angle_constraints =[(0,1,2),(0,2,1),(1,0,2),(1,3,5),(2,4,6),(5,7,9),(6,8,10),(11,13,15),(12,14,16)]

        # 从热图中解码出关节点坐标
        pred_coords = self.decode_heatmap_to_coords(pred_heatmaps)
        target_coords = self.decode_heatmap_to_coords(target_heatmaps)

        # 计算预测和真实的肢体长度和角度
        pred_limb_lengths = torch.zeros(B, len(limb_pairs)).to(pred_heatmaps.device)
        target_limb_lengths = torch.zeros(B, len(limb_pairs)).to(pred_heatmaps.device)
        pred_angles = torch.zeros(B, len(angle_constraints) - 1).to(pred_heatmaps.device)
        target_angles = torch.zeros(B, len(angle_constraints) - 1).to(pred_heatmaps.device)


        for i, (j1, j2) in enumerate(limb_pairs):
            # 计算预测的肢体长度
            pred_limb_lengths[:, i] = torch.norm(pred_coords[:, j1] - pred_coords[:, j2], dim=-1)
            target_limb_lengths[:, i] = torch.norm(target_coords[:, j1] - target_coords[:, j2], dim=-1)


        # std_length =  torch.mean(target_limb_lengths, dim=-1).unsqueeze(-1)
        std_length = torch.std(target_limb_lengths, dim=-1, unbiased=False).unsqueeze(-1)

        limb_length_loss = torch.exp(
            -1.0 * (pred_limb_lengths - target_limb_lengths) ** 2 / (2 * std_length ** 2  + 1e-9)
        )
        limb_length_loss = -torch.log(torch.mean(limb_length_loss, dim=-1) + 1e-9)

            # 2. 肢体角度的余弦约束

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
        """
            F: [B,192,1024]
            Y: [B,17,64,48]
        """
        batch_size = Feat.shape[0]
        temperature = 0.05
        Feat = Feat.permute(0, 2, 1).reshape(batch_size, -1, 16, 12).contiguous()

        for layer in model.backbone.deconv_layers:
            Feat = layer(Feat)
        pred = model.backbone.final_layer(Feat)

        pred = pred.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        Y = Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        mi = kl_div(input=self.softmax(pred.detach() / temperature), target=self.softmax(Y / temperature),
                    reduction='mean')  # pixel-level

        return mi

    def feat_label_mi_estimation_V2(self, Y, Feat, model):
        """
            F: [B,128,64,48]
            Y: [B,17,64,48]
        """
        batch_size = Feat.shape[0]
        temperature = 0.05

        pred = model.head[-1](Feat)

        pred = pred.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        Y = Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        mi = kl_div(input=self.softmax(pred.detach() / temperature), target=self.softmax(Y / temperature),
                    reduction='mean')  # pixel-level

        return mi

    def label_label_mi_estimation(self, Y, pred_Y):
        """
            F: [B,17,64,48]
            Y: [B,17,64,48]
        """
        batch_size = pred_Y.shape[0]
        temperature = 0.05
        # pred_Y = self.hrnet.final_layer(Feat)  # B,48,96,72 -> B,17,96,72
        pred_Y = pred_Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        Y = Y.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        mi = kl_div(input=self.softmax(pred_Y.detach() / temperature), target=self.softmax(Y / temperature),
                    reduction='mean')

        return mi

    def feat_feat_mi_estimation(self, F1, F2):
        """
            F1: [B,192,1024]
            F2: [B,192,1024]
        """
        batch_size, N = F1.shape[0], F1.shape[1]
        temperature = 0.05
        F1 = F1.reshape(batch_size * N, -1)
        F2 = F2.reshape(batch_size * N, -1)
        mi = kl_div(input=self.softmax(F1.detach() / temperature), target=self.softmax(F2 / temperature),
                    reduction='mean')

        return mi

    def feat_label_mi_estimation_keypoint_gt(self, gt , kpt_tokens):

        batch_size = gt.shape[0]
        temperature = 0.05

        pred = self.mlp(kpt_tokens)


        pred = pred.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        Y = gt.reshape(batch_size, 17, -1).reshape(batch_size * 17, -1)
        mi = kl_div(input=self.softmax(pred.detach() / temperature), target=self.softmax(Y / temperature),
                    reduction='mean')  # pixel-level

        return mi

    def eval(self, model, dataloader, tb_writer_dict, **kwargs):

        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time, data_time, losses, acc ,acc_backbone = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()


        phase = kwargs.get("phase", VAL_PHASE)
        epoch = kwargs.get("epoch", "specified_model")
        # switch to evaluate mode
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        # prepare data fro validate
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=float)
        all_preds_backbone = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=float)
        all_boxes = np.zeros((num_samples, 6))
        all_boxes_backbone = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        filenames_map = {}
        filenames_counter = 0
        imgnums = []
        idx = 0
        idx_backbone = 0
        acc_threshold = 0.7

        ###
        result_output_dir, vis_output_dir = self.vis_setup(logger, phase, epoch)
        ###

        with torch.no_grad():
            end = time.time()
            num_batch = len(dataloader)
            for iter_step in range(self.max_iter_num):
                input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight,  meta = next(self.dataloader_iter)

                if phase == VAL_PHASE:
                    self._before_val_iter(input_x)

                data_time.update(time.time() - end)
                margin_left = meta["margin_left"]
                margin_right = meta["margin_right"]
                margin = torch.stack([margin_left, margin_right], dim=1).cuda()
                target_heatmaps = target_heatmaps.cuda(non_blocking=True)

                # outputs = model(concat_input, margin)
                concat_input = torch.cat((input_x, input_sup_A, input_sup_B), 1).cuda()

                outputs , htms_backbone ,feat_curr , fused_feature , kpt_tokens_curr = model(concat_input.cuda(), margin = margin)





                if phase == VAL_PHASE:
                    if isinstance(model, torch.nn.DataParallel):
                        vis_dict = getattr(model.module, "vis_dict", None)
                    else:
                        vis_dict = getattr(model, "vis_dict", None)

                    if vis_dict:
                        self._running_val_iter(vis_dict=vis_dict, model_input=[input_x, input_sup_A, input_sup_B])

                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    pred_heatmaps = outputs[-1]
                else:
                    pred_heatmaps = outputs

                FLIP_TEST = self.cfg.VAL.FLIP_VAL
                if FLIP_TEST:
                    concat_input_flipped = concat_input.flip(3)

                    pred_heatmaps_flipped , htms_backbone_flipped ,feat_curr , fused_feature , kpt_tokens_curr= model(concat_input_flipped.cuda(),
                                                                    margin = margin)
                    from datasets.process.pose_process import flip_back
                    pred_heatmaps_flipped = flip_back(pred_heatmaps_flipped.cpu().numpy(),
                                                      dataset.flip_pairs)
                    htms_backbone_flipped = flip_back(htms_backbone_flipped.cpu().numpy(), dataset.flip_pairs)

                    pred_heatmaps_flipped = torch.from_numpy(pred_heatmaps_flipped.copy()).cuda()
                    htms_backbone_flipped = torch.from_numpy(htms_backbone_flipped.copy()).cuda()


                    pred_heatmaps_flipped[:, :, :, 1:] = pred_heatmaps_flipped.clone()[:, :, :,
                                                         0:-1]
                    htms_backbone_flipped[:, :, :, 1:] = htms_backbone_flipped.clone()[:, :, :, 0:-1]
                    pred_heatmaps = (pred_heatmaps + pred_heatmaps_flipped) * 0.5
                    htms_backbone = (htms_backbone + htms_backbone_flipped) * 0.5


                _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())

                acc.update(avg_acc, cnt)

                _, avg_acc_backbone, cnt_backbone, _ = accuracy(htms_backbone.detach().cpu().numpy(),
                                              target_heatmaps.detach().cpu().numpy())

                acc_backbone.update(avg_acc_backbone, cnt_backbone)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    msg = '{}: [{}/{}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t   \
                          {acc_backbone.val:.3f} ({acc_backbone.avg:.3f})'.format(phase, iter_step, num_batch, batch_time=batch_time,
                                                                          data_time=data_time, acc=acc, acc_backbone=acc_backbone)
                    logger.info(msg)



                #### for eval ####
                for ff in range(len(meta['image'])):
                    cur_nm = meta['image'][ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = meta['center'].numpy()
                scale = meta['scale'].numpy()
                score = meta['score'].numpy()
                num_images = input_x.size(0)

                preds, maxvals = dark_get_final_preds(pred_heatmaps.clone().cpu().numpy(), center, scale)
                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                idx += num_images

                preds_backbone, maxvals_backbone = dark_get_final_preds(htms_backbone.clone().cpu().numpy(), center, scale)
                all_preds_backbone[idx_backbone:idx_backbone + num_images, :, 0:2] = preds_backbone[:, :, 0:2]
                all_preds_backbone[idx_backbone:idx_backbone + num_images, :, 2:3] = maxvals_backbone
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 0:2] = center[:, 0:2]
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 2:4] = scale[:, 0:2]
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 5] = score
                idx_backbone += num_images









                # tensorboard writ
                self.global_steps += 1
                #
                self._after_val_iter(meta["image"], preds, maxvals, vis_output_dir, center, scale)


        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))

        name_values, perf_indicator = dataset.evaluate(self.cfg, all_preds, result_output_dir, all_boxes,
                                                       filenames_map, filenames, imgnums)

        name_values_backbone, perf_indicator_backbone = dataset.evaluate(self.cfg, all_preds_backbone, result_output_dir, all_boxes_backbone,
                                                       filenames_map, filenames, imgnums)

        self._print_name_value(name_values_backbone, 'ViT_Backbone')

        model_name = self.cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, model_name)
        else:
            self._print_name_value(name_values, model_name)

        tb_writer_dict["global_steps"] = self.global_steps

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

    def _before_val_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "val_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _running_val_iter(self, **kwargs):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        vis_dict = kwargs.get("vis_dict")
        #
        show_image_num = min(3, len(vis_dict["current_x"]))
        current_x = vis_dict["current_x"][0:show_image_num]  # [N,3,384,288]
        previous_x = vis_dict["previous_x"][0:show_image_num]  # [N,3,384,288]
        next_x = vis_dict["next_x"][0:show_image_num]  # [N,3,384,288]
        current_rough_heatmaps = vis_dict["current_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        previous_rough_heatmaps = vis_dict["previous_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        next_rough_heatmaps = vis_dict["next_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        diff_A = vis_dict["diff_A"][0:show_image_num]  # [N,17,96,72]
        diff_B = vis_dict["diff_B"][0:show_image_num]  # [N,17,96,72]
        diff_heatmaps = vis_dict["diff_heatmaps"][0:show_image_num]  # [N,34,96,72]
        support_heatmaps = vis_dict["support_heatmaps"][0:show_image_num]  # [N,17,96,72]
        prf_ptm_combine_featuremaps = vis_dict["prf_ptm_combine_featuremaps"][0:show_image_num]  # [N,96,96,72]
        warped_heatmaps_list = [warped_heatmaps[0:show_image_num] for warped_heatmaps in vis_dict["warped_heatmaps_list"]]  # [N,17,96,72]
        output_heatmaps = vis_dict["output_heatmaps"][0:show_image_num]  # [N,17,96,72]

        show_three_input_image = make_grid(reverse_transforms(torch.cat([previous_x, current_x, next_x], dim=0)), nrow=show_image_num)
        self.tb_writer.add_image('01_three_input_image', show_three_input_image, global_step=self.global_steps)

        # show 2.
        three_rough_heatmaps_channels = []
        current_rough_heatmap_channels = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmap_channels = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmap_channels = next_rough_heatmaps.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            three_rough_heatmaps_channels.append(current_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(previous_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(next_rough_heatmap_channels[i])

        three_heatmaps_tensor = torch.clamp_min(torch.cat(three_rough_heatmaps_channels, dim=0), 0)
        three_heatmaps_image = make_grid(three_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('02_three_heatmaps_image', three_heatmaps_image, global_step=self.global_steps)

        # show 3.
        two_diff_channels = []
        diff_A_channels = diff_A.split(1, dim=1)
        diff_B_channels = diff_B.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            two_diff_channels.append(diff_A_channels[i])
            two_diff_channels.append(diff_B_channels[i])

        two_diff_channels_tensor = torch.clamp_min(torch.cat(two_diff_channels, dim=0), 0)
        two_diff_image = make_grid(two_diff_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('03_two_diff_image', two_diff_image, global_step=self.global_steps)

        # show4.
        diff_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(diff_heatmaps, 1, dim=1), dim=0), 0)
        diff_heatmaps_channels_image = make_grid(diff_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('04_diff_heatmaps_channels_image', diff_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show5.
        support_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(support_heatmaps, 1, dim=1), dim=0), 0)
        support_heatmaps_channels_image = make_grid(support_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('05_support_heatmaps_channels_image', support_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show6.
        prf_ptm_combine_featuremaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(prf_ptm_combine_featuremaps, 1, dim=1), dim=0),
                                                                      0)
        prf_ptm_combine_featuremaps_channels_image = make_grid(prf_ptm_combine_featuremaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('06_prf_ptm_combine_featuremaps_channels_image',
                                 prf_ptm_combine_featuremaps_channels_image, global_step=self.global_steps)

        # show7.
        warped_heatmaps_1_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[0], 1, dim=1), dim=0), 0)
        warped_heatmaps_1_channels_image = make_grid(warped_heatmaps_1_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('07_warped_heatmaps_1_channels_image', warped_heatmaps_1_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_2_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[1], 1, dim=1), dim=0), 0)
        warped_heatmaps_2_channels_image = make_grid(warped_heatmaps_2_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('08_warped_heatmaps_2_channels_image', warped_heatmaps_2_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_3_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[2], 1, dim=1), dim=0), 0)
        warped_heatmaps_3_channels_image = make_grid(warped_heatmaps_3_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('09_warped_heatmaps_3_channels_image', warped_heatmaps_3_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_4_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[3], 1, dim=1), dim=0), 0)
        warped_heatmaps_4_channels_image = make_grid(warped_heatmaps_4_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('10_warped_heatmaps_4_channels_image', warped_heatmaps_4_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_5_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[4], 1, dim=1), dim=0), 0)
        warped_heatmaps_5_channels_image = make_grid(warped_heatmaps_5_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('11_warped_heatmaps_5_channels_image', warped_heatmaps_5_channels_image,
                                 global_step=self.global_steps)

        # show8.
        output_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(output_heatmaps, 1, dim=1), dim=0), 0)
        output_heatmaps_channels_image = make_grid(output_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('12_output_heatmaps_channels_image', output_heatmaps_channels_image,
                                 global_step=self.global_steps)

    def _after_val_iter(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
        cfg = self.cfg
        # prepare data
        coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
        bboxes = []
        for index in range(len(center)):
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
            from ..vis_helper import draw_skeleton_in_origin_image
            draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON,
                                          vis_bbox=cfg.DEBUG.VIS_BBOX)

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

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir, "use_gt_box" if self.cfg.VAL.USE_GT_BBOX else "use_precomputed_box")
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
