import cv2
import torch
import numpy as np
import os
from PIL import Image
import shutil
from datasets.transforms.build import reverse_transforms


def dir_judge(folder_path ):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 文件夹不存在，创建文件夹
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' has been created.")
    else:
        # 文件夹存在，检查是否为空
        if os.listdir(folder_path):
            # 文件夹非空，清空文件夹内容
            print(f"Clearing contents of '{folder_path}'...")
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    # 如果是文件，则删除文件
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    # 如果是目录，则删除目录及其内容
                    else:
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"Folder '{folder_path}' has been cleared.")
        else:
            # 文件夹已存在且为空
            print(f"Folder '{folder_path}' already exists and is empty.")


def _normalize_heatmap_tensor(heatmap):
    heatmap = heatmap.detach()
    heatmap_min = torch.min(heatmap)
    heatmap_max = torch.max(heatmap)
    denom = heatmap_max - heatmap_min
    if denom.item() <= 1e-12:
        return torch.zeros_like(heatmap)
    return (heatmap - heatmap_min) / denom


def _sum_valid_heatmaps(heatmaps, skip_joints):
    valid_indices = [j for j in range(heatmaps.shape[0]) if j not in skip_joints]
    if not valid_indices:
        return np.zeros(heatmaps.shape[-2:], dtype=np.float32)
    return np.sum(heatmaps[valid_indices].cpu().detach().numpy(), axis=0)


def _heatmap_sample_dir(output_root, image_paths, index, batch_id):
    if not image_paths:
        return os.path.join(output_root, str(batch_id), str(index))

    image_path = image_paths[index]
    path_parts = image_path.replace("\\", "/").split("/")
    if "images" in path_parts:
        rel_parts = path_parts[path_parts.index("images") + 1:]
    else:
        rel_parts = path_parts[-2:]

    frame_name = os.path.splitext(rel_parts[-1])[0]
    video_parts = rel_parts[:-1]
    return os.path.join(output_root, *video_parts, frame_name, f"person_{batch_id}_{index}")


def save_heatmaps(dict, image, target, batch_id, save_dir=None, skip_joints=None, image_paths=None):
    skip_joints = set(skip_joints or [])
    output_root = os.path.join(save_dir, "heatmaps") if save_dir else f'./output/heatmaps/{batch_id}'
    os.makedirs(output_root, exist_ok=True)
    batch_total = target.shape[0]
    htms_num = target.shape[1]

    H, W = image[0].shape[-2], image[0].shape[-1]
    image_reverse_t = reverse_transforms(image[0]).detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    image_reverse_left = reverse_transforms(image[1]).detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    image_reverse_right = reverse_transforms(image[2]).detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)


    final_htms = dict['htms_out']
    diff_f = dict["diff_f"]
    diff_b = dict["diff_b"]

    for i in range(0, batch_total):
        each_dir = _heatmap_sample_dir(output_root, image_paths, i, batch_id)
        final_htms_dir = each_dir + '/htms_out'
        target_dir = each_dir + '/target'
        dir_judge(final_htms_dir)
        dir_judge(target_dir)

        dir_list = [final_htms_dir, target_dir]

        image_t = image_reverse_t[i]
        image_left = image_reverse_left[i]
        image_right = image_reverse_right[i]


        final_htms_each = final_htms[i]
        target_each = target[i]
        diff_f_each = diff_f[i]
        diff_b_each = diff_b[i]

        htms_list = [ final_htms_each, target_each]
        image_list = [image_t, image_t, image_t]

        cv2.imwrite(os.path.join(each_dir, f'image_t.png'), image_t)
        cv2.imwrite(os.path.join(each_dir, f'image_left.png'), image_left)
        cv2.imwrite(os.path.join(each_dir, f'image_right.png'), image_right)


        for index, x in enumerate(htms_list):
            for j in range(htms_num):
                if j in skip_joints:
                    continue
                heatmap = x[j]

                heatmap_normalized = _normalize_heatmap_tensor(heatmap)

                # 应用颜色映射，这里使用 'jet' 颜色映射
                heatmap_colormapped = cv2.applyColorMap(np.uint8(255 * heatmap_normalized.cpu().numpy()),
                                                        cv2.COLORMAP_JET)

                heatmap_colormapped = cv2.resize(heatmap_colormapped, (W, H), interpolation=cv2.INTER_LINEAR)

                alpha = 0.5
                composite_image = cv2.addWeighted(image_list[index], 1 - alpha, heatmap_colormapped, alpha, 0)

                cv2.imwrite(os.path.join(dir_list[index], f'heatmap_{j}.png'), composite_image)

        #这里是保存17张热图加在一起的热图。
        # 假设 heatmaps 是一个包含 17 个热图的列表
        combined_heatmap = _sum_valid_heatmaps(final_htms_each, skip_joints)

        # 归一化到 0-255 范围
        normalized_combined_heatmap = cv2.normalize(combined_heatmap, None, 0, 255, cv2.NORM_MINMAX)

        # 转换为彩色热图
        colored_combined_heatmap = cv2.applyColorMap(normalized_combined_heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        colormapped_combined_heatmap = cv2.resize(colored_combined_heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

        alpha = 0.7
        composite_combined_heatmap = cv2.addWeighted(image_t, 1 - alpha, colormapped_combined_heatmap, alpha, 0)

        cv2.imwrite(os.path.join(each_dir, f'combined_final_heatmap.png'), composite_combined_heatmap)

        #下面是保存diff前向和后向的图
        #---------------前向----------------
        combined_diff_f = _sum_valid_heatmaps(diff_f_each, skip_joints)

        # 归一化到 0-255 范围
        normalized_combined_diff_f = cv2.normalize(combined_diff_f, None, 0, 255, cv2.NORM_MINMAX)

        # 转换为彩色热图
        colored_combined_diff_f = cv2.applyColorMap(normalized_combined_diff_f.astype(np.uint8), cv2.COLORMAP_JET)
        colormapped_combined_diff_f = cv2.resize(colored_combined_diff_f, (W, H), interpolation=cv2.INTER_LINEAR)

        alpha = 0.7
        composite_combined_diff_f = cv2.addWeighted(image_t, 1 - alpha, colormapped_combined_diff_f, alpha, 0)

        cv2.imwrite(os.path.join(each_dir, f'combined_diff_f.png'), composite_combined_diff_f)
        # ---------------后向----------------
        combined_diff_b = _sum_valid_heatmaps(diff_b_each, skip_joints)

        # 归一化到 0-255 范围
        normalized_combined_diff_b = cv2.normalize(combined_diff_b, None, 0, 255, cv2.NORM_MINMAX)

        # 转换为彩色热图
        colored_combined_diff_b = cv2.applyColorMap(normalized_combined_diff_b.astype(np.uint8), cv2.COLORMAP_JET)
        colormapped_combined_diff_b = cv2.resize(colored_combined_diff_b, (W, H), interpolation=cv2.INTER_LINEAR)

        alpha = 0.7
        composite_combined_diff_b = cv2.addWeighted(image_t, 1 - alpha, colormapped_combined_diff_b, alpha, 0)

        cv2.imwrite(os.path.join(each_dir, f'combined_diff_b.png'), composite_combined_diff_b)
        #------------target也combine起来---------
        combined_target = _sum_valid_heatmaps(target_each, skip_joints)

        # 归一化到 0-255 范围
        normalized_combined_target = cv2.normalize(combined_target, None, 0, 255, cv2.NORM_MINMAX)

        # 转换为彩色热图
        colored_combined_target = cv2.applyColorMap(normalized_combined_target.astype(np.uint8), cv2.COLORMAP_JET)
        colormapped_combined_target = cv2.resize(colored_combined_target, (W, H), interpolation=cv2.INTER_LINEAR)

        alpha = 0.7
        composite_combined_target = cv2.addWeighted(image_t, 1 - alpha, colormapped_combined_target, alpha, 0)

        cv2.imwrite(os.path.join(each_dir, f'combined_target.png'), composite_combined_target)


    print("all vis images have been saved!!!")
