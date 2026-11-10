import torch


def cluster_dpc_knn_dynamic(
    x,
    alpha=0.5,
    min_keep=16,
    max_keep=None,
    k=2,
    token_mask=None,
    pad_to_batch_max=False,
    return_mask=False,
    return_debug=False,
):
    """
    根据 DPC 得分阈值 μ + alpha*σ 自适应选择聚类中心。

    输入：
      x          : [B, N, C]，B 个样本，N 个 token，每个 token 维度为 C
      alpha      : float，阈值系数，越大保留 token 越少
      min_keep   : int，最少保留 token 数
      max_keep   : int|None，最多保留 token 数，None 表示不限
      k          : int，计算密度时使用的近邻数
      token_mask : [B, N]，可选，1 表示有效 token，0 表示忽略
    输出：
      index_down : [B, K]，每个样本选出的 K 个聚类中心的 token 索引
      idx_cluster: [B, N]，每个 token 所属聚类中心的编号（0~K-1）
      valid_mask : [B, K]，pad_to_batch_max=True 时返回，1 表示真实保留 token
    """
    with torch.no_grad():
        bsz, token_num, dim = x.shape  # B, N, C

        # dist_matrix: [B, N, N]，归一化欧氏距离矩阵
        dist_matrix = torch.cdist(x, x) / (dim ** 0.5)

        # 若提供掩码，将掩码外 token 的距离设为极大值，使其不参与聚类
        if token_mask is not None:
            token_mask = token_mask > 0                          # [B, N] bool
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # dist_nearest: [B, N, k]，每个 token 的 k 个最近邻距离
        # density:      [B, N]，局部密度（越大表示周围 token 越密集）
        dist_nearest, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # Break equal-density ties deterministically by token index.
        tie_break = torch.arange(token_num, device=x.device, dtype=density.dtype).unsqueeze(0)
        density = density + tie_break * 1e-12

        if token_mask is not None:
            density = density * token_mask                       # [B, N]

        # mask:     [B, N, N] bool，mask[b,i,j]=1 表示 j 的密度高于 i
        # dist_max: [B, 1, 1]，每个样本的距离最大值，用于填充低密度方向
        # dist:     [B, N]，每个 token 到密度更高邻居的最短距离
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]  # [B, 1, 1]
        dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)  # [B, N]

        # score: [B, N]，DPC 得分 = 密度 × 到高密度邻居的距离
        score = dist * density

        # mu, sigma, threshold: [B, 1]，对每个样本单独计算统计量
        # keep_counts:          [B]，每个样本中得分超过阈值的 token 数量
        # cluster_num:          int，固定长度批次的平均保留数，限制在 [min_keep, max_keep]
        mu = score.mean(dim=-1, keepdim=True)                    # [B, 1]
        sigma = score.std(dim=-1, unbiased=False, keepdim=True)  # [B, 1]
        threshold = mu + alpha * sigma                           # [B, 1]
        raw_keep_counts = (score > threshold).sum(dim=-1).to(torch.long)  # [B]

        min_keep = max(1, int(min_keep))
        max_keep = token_num if max_keep is None else int(max_keep)
        max_keep = max(min_keep, min(max_keep, token_num))
        keep_counts = raw_keep_counts.clamp(min=min_keep, max=max_keep)  # [B]

        if pad_to_batch_max:
            cluster_num = int(keep_counts.max().item())
            index_down = torch.zeros(bsz, cluster_num, device=x.device, dtype=torch.long)
            valid_mask = torch.zeros(bsz, cluster_num, device=x.device, dtype=torch.bool)
            for b in range(bsz):
                keep_b = int(keep_counts[b].item())
                _, index_b = torch.topk(score[b], k=keep_b, dim=-1)
                index_b, _ = torch.sort(index_b, dim=-1)
                index_down[b, :keep_b] = index_b
                valid_mask[b, :keep_b] = True
        else:
            cluster_num = int(torch.round(keep_counts.float().mean()).item())  # scalar int
            cluster_num = max(min_keep, min(cluster_num, max_keep))
            valid_mask = None

            # index_down: [B, cluster_num]，得分最高的 cluster_num 个 token 的索引（已排序）
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)
            index_down, _ = torch.sort(index_down, dim=-1)           # [B, cluster_num]

        # idx_cluster:   [B, N]，每个 token 所属聚类中心编号
        # selected_dist: [N, cluster_num]，当前样本所有 token 到各聚类中心的距离
        idx_cluster = torch.zeros(bsz, token_num, device=x.device, dtype=torch.long)
        for b in range(bsz):
            selected_num = int(valid_mask[b].sum().item()) if valid_mask is not None else cluster_num
            selected_dist = dist_matrix[b, :, index_down[b, :selected_num]]    # [N, selected_num]
            idx_cluster[b] = selected_dist.argmin(dim=-1)        # [N]
            # 确保聚类中心自身归属编号正确（覆盖 argmin 可能的误分配）
            idx_cluster[b, index_down[b, :selected_num]] = torch.arange(selected_num, device=x.device)

    if return_debug:
        debug = {
            "score": score.detach(),
            "threshold": threshold.detach(),
            "score_mu": mu.squeeze(-1).detach(),
            "score_sigma": sigma.squeeze(-1).detach(),
            "candidate_keep_raw": raw_keep_counts.detach(),
            "keep_counts": keep_counts.detach(),
            "candidate_keep_clipped": keep_counts.detach(),
            "final_cluster_num": torch.tensor(cluster_num, device=x.device, dtype=torch.long),
            "valid_mask": valid_mask.detach() if valid_mask is not None else None,
            "clip_low_mask": (raw_keep_counts < min_keep).detach(),
            "clip_high_mask": (raw_keep_counts > max_keep).detach(),
        }
        if return_mask:
            return index_down, idx_cluster, valid_mask, debug
        return index_down, idx_cluster, debug

    if return_mask:
        return index_down, idx_cluster, valid_mask

    return index_down, idx_cluster
