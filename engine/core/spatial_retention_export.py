"""Export coarse-branch token selections as masks on the original grid.

Accepts model.coarse_prune_vis or a binary keep_mask (True = retained).
Arrays are stored with NumPy; Torch tensors are detached and converted to arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def records_from_collated_meta(meta):
    """Build records from PoseTrack metadata collated by a PyTorch DataLoader.

    Metadata must include dataset_index. Frame order follows the model input:
    the current image followed by ref_sup_image/supp_idx. Custom collation or
    reordered inputs require a corresponding metadata adapter.
    """
    sample_ids = _array(meta["dataset_index"])
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in "iu":
        raise ValueError("Expected collated integer dataset_index [B]")
    batch_size = len(sample_ids)
    images = meta["image"]
    current = _array(meta["frame_id"]["current_idx"])
    support = meta["frame_id"]["supp_idx"]
    if not isinstance(support, (list, tuple)):
        raise ValueError("Expected default-collated supp_idx: list of [B] tensors")
    frames = [current] + [_array(v) for v in support]
    if any(v.shape != (batch_size,) or v.dtype.kind not in "iu" for v in frames):
        raise ValueError("Frame IDs must be integer [B] arrays in input order")
    if isinstance(images, str) or len(images) != batch_size:
        raise ValueError("Expected one image path per batch row")
    extras = {}
    for key in ("center", "scale", "rotation", "joints", "joints_vis"):
        if key in meta:
            value = _array(meta[key])
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(f"Expected batch-first metadata: {key}")
            extras[key] = value
    records = []
    for b, sample_id in enumerate(sample_ids):
        image = str(images[b])
        # Full parent path avoids merging different videos with the same basename.
        normalized = image.replace("\\", "/")
        if "/" not in normalized:
            raise ValueError("Image path must include its video directory")
        record = dict(sample_id=str(int(sample_id)), image=image,
                      video_id=normalized.rsplit("/", 1)[0],
                      frame_ids=[int(frame[b]) for frame in frames])
        record.update({key: value[b].tolist() for key, value in extras.items()})
        records.append(record)
    return records


def selection_mask(vis, *, padding="masked"):
    """Return a [B,T,H,W] boolean mask in model input order.

    padding='masked' requires valid_mask for selected indices in padded batches.
    padding='none' accepts batches whose selected indices are all valid.
    A binary keep_mask can directly supply the original-grid selection as
    [B,T*H*W] or [B,T,H,W], with 1/True meaning retained. Selection counts come
    from the mask or the valid selected indices.
    """
    if vis is None:
        raise ValueError("Missing coarse_prune_vis; enable collect_prune_vis first")
    if padding not in ("masked", "none"):
        raise ValueError("padding must be 'masked' or 'none'")
    if vis.get("branch", "coarse") != "coarse":
        raise ValueError("Expected ISP/coarse payload, not ATP/fine")
    shape = tuple(int(vis[k]) for k in ("frame_num", "grid_h", "grid_w"))
    if min(shape) < 1:
        raise ValueError("Invalid token grid")
    total = int(np.prod(shape))
    if int(vis["total_tokens"]) != total:
        raise ValueError("T*H*W differs from total_tokens")
    if "tokens_per_frame" in vis and int(vis["tokens_per_frame"]) != shape[1] * shape[2]:
        raise ValueError("H*W differs from tokens_per_frame")
    if "keep_mask" in vis:
        direct = _array(vis["keep_mask"])
        if (direct.ndim not in (2, 4) or direct.shape[0] == 0
                or (direct.ndim == 2 and direct.shape[1] != total)
                or (direct.ndim == 4 and direct.shape[1:] != shape)):
            raise ValueError("keep_mask must be [B,T*H*W] or [B,T,H,W]")
        if not np.isin(direct, [0, 1]).all():
            raise ValueError("Use the FINAL binary keep mask, not scores or attention weights")
        direct = direct.astype(bool).reshape((-1,) + shape)
        if np.any(direct.reshape(len(direct), -1).sum(1) == 0):
            raise ValueError("Empty final keep mask")
        if "keep_index" in vis:
            indexed = dict(vis)
            del indexed["keep_mask"]
            if not np.array_equal(direct, selection_mask(indexed, padding=padding)):
                raise ValueError("Direct keep mask disagrees with valid retained indices")
        return direct
    index = _array(vis["keep_index"])
    if index.ndim != 2 or index.shape[0] == 0 or index.dtype.kind not in "iu":
        raise ValueError("keep_index must be a nonempty [B,K_pad] integer array")
    valid = vis.get("valid_mask")
    if valid is None:
        if padding == "masked":
            raise ValueError("valid_mask missing; do not count padding as retained tokens")
        valid = np.ones(index.shape, dtype=bool)
    else:
        valid = _array(valid)
        if valid.shape != index.shape or not np.isin(valid, [0, 1]).all():
            raise ValueError("Invalid selected-token valid_mask")
        valid = valid.astype(bool)
        if padding == "none" and not valid.all():
            raise ValueError("Payload is padded but padding='none' was requested")
    mask = np.zeros((len(index), total), dtype=bool)
    for b in range(len(index)):
        kept = index[b, valid[b]]
        if len(kept) == 0 or np.any(kept < 0) or np.any(kept >= total):
            raise ValueError("Empty selection or out-of-range valid token index")
        if len(np.unique(kept)) != len(kept):
            raise ValueError("Duplicate VALID indices; cannot infer an actual budget")
        mask[b, kept] = True
    return mask.reshape((len(index),) + shape)


class SpatialRetentionExporter:
    """One independent single-process recorder per fresh output directory.

    records: one dict per batch row, with unique sample_id, video_id, image,
    and frame_ids in model INPUT order (duplicates at video ends are allowed).
    Additional JSON fields, e.g. crop center/scale, are preserved unchanged.
    """

    def __init__(self, output_dir, *, run_metadata, input_frame_offsets, padding="masked"):
        required = ("dataset", "split", "checkpoint", "checkpoint_sha256", "config_sha256",
                    "source_sha256", "dynamic_alpha", "dynamic_min_keep_coarse",
                    "dynamic_max_keep_coarse", "eval_mode", "augmentation")
        missing = [k for k in required if k not in run_metadata]
        if missing:
            raise ValueError(f"Missing provenance fields: {missing}")
        if run_metadata["split"] not in ("val", "validate", "validation") or run_metadata["eval_mode"] is not True:
            raise ValueError("This recorder is for validation in model.eval() mode")
        if not run_metadata["source_sha256"]:
            raise ValueError("Record hashes of actual model/pruning source files")
        if padding not in ("masked", "none"):
            raise ValueError("Invalid padding mode")
        self.offsets = np.asarray(input_frame_offsets, dtype=np.int64)
        if self.offsets.ndim != 1 or len(np.unique(self.offsets)) != len(self.offsets) or 0 not in self.offsets:
            raise ValueError("Supply distinct nominal input offsets including target frame 0")
        self.run = json.loads(json.dumps(run_metadata, allow_nan=False))
        self.minimum = int(self.run["dynamic_min_keep_coarse"])
        self.maximum = self.run["dynamic_max_keep_coarse"]
        if self.minimum < 1 or (self.maximum is not None and int(self.maximum) < self.minimum):
            raise ValueError("Invalid configured bounds")
        self.padding = padding
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=False)  # Never mix/overwrite runs.
        self.n = 0
        self.shards = 0
        self.shape = None
        self.keep_sum = None
        self.ids = set()
        self.by_video = {}
        self.closed = False
        self.keep_min = None
        self.keep_max = None
        meta = {"schema": "isp_spatial_retention_v1", "run": self.run,
                "input_frame_offsets": self.offsets.tolist(), "padding_mode": padding,
                "flatten_order": "frame-major, row-major: t*(H*W)+y*W+x",
                "packed_bitorder": "little", "statistical_unit": "person-centered input clip",
                "weighting": "equal per input clip; not equal per batch",
                "baseline": "uniform random selection with the same per-clip K from T*H*W tokens"}
        with (self.out / "run_metadata.json").open("x", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, allow_nan=False)

    def add_batch(self, vis, records):
        if self.closed:
            raise ValueError("Recorder already finished")
        masks = selection_mask(vis, padding=self.padding)
        batch_size, *dims = masks.shape
        shape = tuple(dims)
        if len(self.offsets) != shape[0]:
            raise ValueError("Input frame order length differs from frame_num")
        if self.shape is not None and self.shape != shape:
            raise ValueError("Grid changed within a run")
        if len(records) != batch_size:
            raise ValueError("Need exactly one metadata record per input clip")
        records = json.loads(json.dumps(records, allow_nan=False))
        ids = []
        for rec in records:
            for key in ("sample_id", "video_id", "image"):
                if not isinstance(rec.get(key), str) or not rec[key]:
                    raise ValueError(f"Nonempty string required: {key}")
            if not isinstance(rec.get("frame_ids"), list) or len(rec["frame_ids"]) != shape[0]:
                raise ValueError("Record the actual frame IDs in model input order")
            ids.append(rec["sample_id"])
        if len(set(ids)) != len(ids) or self.ids.intersection(ids):
            raise ValueError("Duplicate sample_id; check sampler, flip TTA or repeated logging")
        total = int(np.prod(shape))
        counts = masks.reshape(batch_size, -1).sum(1).astype(np.int32)
        maximum = total if self.maximum is None else min(total, int(self.maximum))
        if np.any(counts < self.minimum) or np.any(counts > maximum):
            raise ValueError("ACTUAL selected counts violate recorded run bounds; no clipping applied")
        for rec, count in zip(records, counts):
            rec.update(actual_keep_count=int(count), total_tokens=total,
                       pruning_rate=1 - int(count) / total)
        packed = np.packbits(masks.reshape(batch_size, -1), axis=1, bitorder="little")
        path = self.out / f"batch_{self.shards:06d}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, packed_keep_mask=packed, grid_shape=np.asarray(shape),
                                keep_counts=counts,
                                records_json=np.asarray([json.dumps(r, ensure_ascii=False) for r in records]))
        if self.shape is None:
            self.shape = shape
            self.keep_sum = np.zeros(shape, dtype=np.int64)
        self.keep_sum += masks.sum(axis=0, dtype=np.int64)
        for rec, mask in zip(records, masks):
            video = rec["video_id"]
            if video not in self.by_video:
                self.by_video[video] = [0, np.zeros(shape, dtype=np.int64)]
            self.by_video[video][0] += 1
            self.by_video[video][1] += mask
        self.keep_min = int(counts.min()) if self.keep_min is None else min(self.keep_min, int(counts.min()))
        self.keep_max = int(counts.max()) if self.keep_max is None else max(self.keep_max, int(counts.max()))
        self.n += batch_size
        self.shards += 1
        self.ids.update(ids)

    def finish(self, *, expected_samples):
        if self.closed or self.n == 0:
            raise ValueError("No samples collected or run already finished")
        if self.n != expected_samples:
            raise ValueError(f"Incomplete/duplicated dataset: {self.n} != {expected_samples}")
        p_frame = self.keep_sum / self.n
        p_spatial = p_frame.mean(axis=0)
        baseline = float(self.keep_sum.sum() / (self.n * np.prod(self.shape)))
        delta = p_spatial - baseline
        if abs(float(delta.mean())) > 1e-12:
            raise AssertionError("Spatial excess should average to zero")
        videos = sorted(self.by_video)
        with (self.out / "summary.npz").open("xb") as f:
            np.savez_compressed(
                f, sample_count=self.n, grid_shape=np.asarray(self.shape),
                input_frame_offsets=self.offsets, chronological_order=np.argsort(self.offsets),
                retention_by_frame=p_frame, retention_spatial=p_spatial,
                mean_keep_ratio=baseline, excess_spatial=delta,
                top_profile=delta.mean(axis=0), right_profile=delta.mean(axis=1),
                video_ids=np.asarray(videos),
                video_sample_counts=np.asarray([self.by_video[v][0] for v in videos]),
                video_keep_sums=np.stack([self.by_video[v][1] for v in videos]))
        result = {"status": "complete", "samples": self.n, "batches": self.shards,
                  "videos": len(videos), "grid_shape": self.shape,
                  "actual_keep_range": [self.keep_min, self.keep_max],
                  "mean_keep_ratio": baseline, "mean_pruning_rate": 1 - baseline,
                  "source_run": "run_metadata.json", "summary": "summary.npz"}
        with (self.out / "completion.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        self.closed = True
        return result


def read_shard(path):
    """Load an exported batch without pickle. Reconstruct exact original masks."""
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(v) for v in data["grid_shape"])
        bits = np.unpackbits(data["packed_keep_mask"], axis=1,
                             count=int(np.prod(shape)), bitorder="little")
        masks = bits.reshape((-1,) + shape).astype(bool)
        counts = data["keep_counts"].copy()
        records = [json.loads(v) for v in data["records_json"].tolist()]
    if not np.array_equal(masks.reshape(len(masks), -1).sum(1), counts):
        raise ValueError("Exported masks disagree with keep counts")
    return masks, records
