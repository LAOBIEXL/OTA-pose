"""Opt-in validation adapter for the user's 2026-10 masked pruning code.

No selection, model, training, or budget changes. Only the original unflipped
forward of a full sequential validation pass is recorded. Default is disabled.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, SequentialSampler

if __package__:
    from .spatial_retention_export import (
        SpatialRetentionExporter, records_from_collated_meta, sha256_file,
    )
else:
    from spatial_retention_export import (
        SpatialRetentionExporter, records_from_collated_meta, sha256_file,
    )


def _value(node, key, default=None):
    return node.get(key, default) if isinstance(node, dict) else getattr(node, key, default)


def _resolve_checkpoint(path, cfg):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(_value(cfg, "ROOT_DIR", ".")) / path
    return path.resolve(strict=True)


class SpatialCaptureSession:
    """Adapter for a checked sequential DataLoader; IDs equal dataset row indices."""

    def __init__(self, model, cfg, dataloader, *, phase, max_validate_batches,
                 evaluation_source, environ=None):
        env = os.environ if environ is None else environ
        output = env.get("OTA_SPATIAL_EXPORT_DIR", "").strip()
        if not output:
            raise ValueError("Set OTA_SPATIAL_EXPORT_DIR to a NEW output directory")
        if str(phase).lower() not in ("val", "validate", "validation"):
            raise ValueError("Spatial capture is for validation, not training/test")
        if max_validate_batches > 0:
            raise ValueError("Capture needs the full validation set; max_validate_batches must be -1")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise ValueError("Use a separate single-process run, not DDP, for spatial capture")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            raise ValueError("DDP capture is not supported")
        if isinstance(model, torch.nn.DataParallel) and len(model.device_ids) > 1:
            raise ValueError("Use exactly one GPU; DataParallel replica attributes are not gathered")
        self.model = model.module if isinstance(model, torch.nn.DataParallel) else model
        if self.model.training:
            raise ValueError("Call model.eval() before creating the recorder")
        if type(dataloader.sampler) is not SequentialSampler:
            raise ValueError("Capture requires the standard SequentialSampler (shuffle=False)")
        if (type(dataloader.batch_sampler) is not BatchSampler
                or dataloader.batch_sampler.sampler is not dataloader.sampler
                or dataloader.drop_last or dataloader.batch_sampler.drop_last):
            raise ValueError("Use standard batching and drop_last=False; custom batching is not supported")
        self.dataset = dataloader.dataset
        if _value(self.dataset, "is_train", False):
            raise ValueError("Do not use a training dataset with random augmentations")
        dataset_phase = _value(self.dataset, "phase", None)
        if dataset_phase is not None and str(dataset_phase).lower() not in ("val", "validate", "validation"):
            raise ValueError("Dataset phase is not validation")
        self.expected = len(self.dataset)
        if self.expected < 1:
            raise ValueError("Empty validation dataset")
        isp = self.model.prune_recover_blocks
        if not isp.use_dynamic_prune or isp.no_prune:
            raise ValueError("Enable the existing dynamic pruning model, with NO_PRUNE=False")
        if getattr(isp, "prune_selection", "score") != "score":
            raise ValueError("Expected the score-based model, not a random-pruning ablation")

        dataset_name = env.get("OTA_SPATIAL_DATASET", "").strip()
        if not dataset_name:
            raise ValueError("Set OTA_SPATIAL_DATASET, e.g. PoseTrack2018; version is not guessed")
        offsets_text = env.get("OTA_SPATIAL_INPUT_OFFSETS", "").strip()
        if not offsets_text:
            raise ValueError("Declare input order with OTA_SPATIAL_INPUT_OFFSETS, e.g. 0,-2,-1,1,2")
        offsets = [int(v.strip()) for v in offsets_text.split(",")]
        if (len(offsets) != int(self.model.frame_num) or len(set(offsets)) != len(offsets)
                or offsets[0] != 0):
            raise ValueError("Offsets must match frame_num, be distinct, and start with current frame 0")
        self.offsets = np.asarray(offsets)

        checkpoint = env.get("OTA_SPATIAL_CHECKPOINT", "").strip()
        if not checkpoint:
            raise ValueError("Set OTA_SPATIAL_CHECKPOINT to the exact checkpoint loaded by this evaluation")
        checkpoint = _resolve_checkpoint(checkpoint, cfg)
        configured_checkpoint = _value(_value(cfg, "VAL", {}), "MODEL_FILE", "")
        if configured_checkpoint:
            if _resolve_checkpoint(configured_checkpoint, cfg) != checkpoint:
                raise ValueError("OTA_SPATIAL_CHECKPOINT differs from cfg.VAL.MODEL_FILE")

        config_text = cfg.dump() if hasattr(cfg, "dump") else json.dumps(cfg, ensure_ascii=False, sort_keys=True)
        source_paths = {
            "model": inspect.getsourcefile(type(self.model)),
            "isp": inspect.getsourcefile(type(isp)),
            "dataset": inspect.getsourcefile(type(self.dataset)),
            "validation": str(evaluation_source),
            "capture": __file__,
            "exporter": inspect.getsourcefile(SpatialRetentionExporter),
        }
        # This is the sibling actually imported by the inspected ISP implementation.
        if source_paths["isp"]:
            dynamic_source = Path(source_paths["isp"]).with_name("token_prune_dynamic.py")
            if dynamic_source.is_file():
                source_paths["dynamic_pruning"] = str(dynamic_source)
        if any(v is None for v in source_paths.values()):
            raise ValueError("Could not locate model/dataset source for provenance")
        source_hashes = {k: sha256_file(v) for k, v in source_paths.items()}
        metadata = dict(
            dataset=dataset_name, split="validation", checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            checkpoint_provenance="declared loaded checkpoint; also checked against VAL.MODEL_FILE when set",
            config_sha256=hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
            config_hash_scope="effective_config.txt, dumped after runtime configuration overrides",
            source_sha256=source_hashes, source_paths=source_paths,
            dynamic_alpha=float(isp.dynamic_alpha),
            dynamic_min_keep_coarse=int(isp.dynamic_min_keep),
            dynamic_max_keep_coarse=(None if isp.dynamic_max_keep is None else int(isp.dynamic_max_keep)),
            eval_mode=True,
            augmentation="validation dataset; original unflipped forward only; preprocessing unchanged",
            batch_size=dataloader.batch_size, expected_samples=self.expected,
            sampler="SequentialSampler", sample_id_scope="row index in this validation dataset",
            implementation="per-sample K; padded to batch maximum; valid_mask=True means retained",
            frame_slots="current frame followed by support frames in concatenation order",
            torch_version=str(torch.__version__), numpy_version=np.__version__,
            train_single_branch_lrb=bool(getattr(self.model, "train_single_branch_lrb", False)),
            inference_single_branch_lrb=bool(getattr(self.model, "inference_single_branch_lrb", False)),
        )
        self.writer = SpatialRetentionExporter(
            output, run_metadata=metadata, input_frame_offsets=offsets, padding="masked",
        )
        with (self.writer.out / "effective_config.txt").open("x", encoding="utf-8", newline="") as stream:
            stream.write(config_text)

    def add_batch(self, meta, batch_size):
        if self.model.training:
            raise ValueError("Model switched to training during validation capture")
        vis = getattr(self.model, "coarse_prune_vis", None)
        if vis is None:
            raise ValueError("Missing coarse_prune_vis: use collect_prune_vis=True in this forward")
        if int(vis.get("current_frame_index", vis.get("key_frame_idx", -1))) != 0:
            raise ValueError("Model payload does not identify input slot 0 as the current frame")
        order = np.argsort(self.offsets).tolist()
        if "temporal_order" in vis and list(vis["temporal_order"]) != order:
            raise ValueError("Declared offsets disagree with model chronological order")
        if len(meta["image"]) != int(batch_size):
            raise ValueError("Metadata size differs from model input batch size")

        # This is safe ONLY because the exact standard sequential sampler was checked.
        # No dataset edit is needed and metadata passed to the original evaluator is untouched.
        row_start = self.writer.n
        row_ids = np.arange(row_start, row_start + int(batch_size), dtype=np.int64)
        if row_start + int(batch_size) > self.expected:
            raise ValueError("More validation samples than the dataset contains")
        capture_meta = dict(meta)
        if "dataset_index" in capture_meta:
            original = capture_meta["dataset_index"]
            original = original.detach().cpu().numpy() if hasattr(original, "detach") else np.asarray(original)
            capture_meta["dataset_index"] = original
        else:
            capture_meta["dataset_index"] = row_ids
        records = records_from_collated_meta(capture_meta)
        if any(len(rec["frame_ids"]) != len(self.offsets) for rec in records):
            raise ValueError("Metadata support frame count differs from the model input")
        actual_frames = np.asarray([rec["frame_ids"] for rec in records])
        if np.any(np.diff(actual_frames[:, order], axis=1) < 0):
            raise ValueError("Actual input frame IDs are inconsistent with declared chronological order")
        for rec in records:
            image_frame = Path(rec["image"].replace("\\", "/")).stem
            if image_frame.isdigit() and int(image_frame) != rec["frame_ids"][0]:
                raise ValueError("Current image filename differs from current frame ID")
        for field in ("candidate_keep_raw", "candidate_keep_clipped",
                      "temporal_redundancy_adj_chrono_prefusion",
                      "temporal_redundancy_current_support_prefusion"):
            if field in vis and vis[field] is not None:
                values = vis[field]
                values = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
                values = values.reshape(-1)
                if len(values) != len(records):
                    raise ValueError(f"Unexpected batch size in {field}")
                for record, value in zip(records, values):
                    record[field] = value.item()
        self.writer.add_batch(vis, records)

    def finish(self):
        return self.writer.finish(expected_samples=self.expected)
