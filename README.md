# OTA-Pose

## 1. Getting Started

### 1.1 Environment Requirement

We recommend using Python 3.8.20 and PyTorch 2.1.1 with CUDA 11.8.

Create and activate the Conda environment:

```bash
conda create -n OTApose python=3.8.20 -y
conda activate OTApose
```

Install PyTorch and Torchvision:

```bash
pip install torch==2.1.1 torchvision==0.16.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### 1.2 Data Preparation

Before training, please prepare the PoseTrack datasets, supplementary annotation files, and pretrained weights.

**1) PoseTrack Datasets**

Download and extract the PoseTrack2018 and PoseTrack2021 datasets. Create a folder `${DATASET_DIR}` to store both datasets.

The directory structure should look like this:

```text
${DATASET_DIR}/
├── PoseTrack2018/
└── PoseTrack2021/
```

For example, the datasets can be organized as follows:

```text
OTA-pose/
└── Dataset/
    ├── PoseTrack2018/
    └── PoseTrack2021/
```

**2) Supplementary Annotation Files**

Prepare the supplementary annotation files used for training and evaluation. These files are provided by the [DCPose project](https://github.com/Pose-Group/DCPose). Please refer to the [DCPose Installation Guide](https://github.com/Pose-Group/DCPose/blob/main/docs/Installation.md) for instructions on obtaining the supplementary files.

Place the `DcPose_supp_files` directory under the project root:

```text
OTA-pose/
└── DcPose_supp_files/
    ├── posetrack18_json_files/
    ├── posetrack18_annotation_dirs/
    ├── posetrack18_precomputed_boxes/
    ├── posetrack21_json_files/
    ├── posetrack21_annotation_dirs/
    └── posetrack21_precomputed_boxes/
```

**3) Pretrained Weights**

Download the ViTPose-L pretrained weights from the [official ViTPose GitHub repository](https://github.com/ViTAE-Transformer/ViTPose). Please refer to the repository for the pretrained model download links.

The downloaded weights should correspond to the ViTPose-L model trained on COCO. Rename the file to `ViTPose-L-weight.pth` if necessary, and place it under the `pretrained` directory:

```text
OTA-pose/
└── pretrained/
    └── ViTPose-L-weight.pth
```

**4) Configuration Settings**

Before training, modify the dataset and pretrained weight paths in the corresponding YAML configuration files according to your local directory structure.

The main configuration files are:

- `configs/otapose/PE/Base_PoseTrack18_5_frame.yaml`
- `configs/otapose/PE/Base_PoseTrack21_5_frame.yaml`

Please check and update the following configuration entries:

- `DATASET.JSON_DIR`: Path to the supplementary JSON files.
- `DATASET.IMG_DIR`: Path to the training dataset.
- `DATASET.TEST_IMG_DIR`: Path to the validation/test dataset.
- `MODEL.PRETRAINED`: Path to the ViTPose pretrained weights.
- `VAL.ANNOT_DIR` and `VAL.COCO_BBOX_FILE`: Validation annotation and detection box paths.
- `TEST.ANNOT_DIR` and `TEST.COCO_BBOX_FILE`: Test annotation and detection box paths.

For example, if the datasets and supplementary files are placed under the project root, the paths can be configured using relative paths:

```yaml
DATASET:
  JSON_DIR: "./DcPose_supp_files/posetrack18_json_files"
  IMG_DIR: "./Dataset/PoseTrack2018"
  TEST_IMG_DIR: "./Dataset/PoseTrack2018"

MODEL:
  PRETRAINED: "./pretrained/ViTPose-L-weight.pth"
```

The paths above are examples for PoseTrack2018. Please make the corresponding changes for PoseTrack2021.

## 2. Training

We support training on both PoseTrack2018 and PoseTrack2021 datasets.

Before running the commands, replace `/path/to/OTA-pose` with the actual path to your project directory. Set `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` according to the number of available GPUs.

### 2.1 PoseTrack2018

```bash
cd /path/to/OTA-pose

# Communication port (can be changed if occupied)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --use-env \
    --nproc_per_node=2 \
    --master_port=29511 \
    tools/run.py \
    --cfg configs/otapose/PE/posetrack18/no_supp_targ_5_frame.yaml \
    --root_dir /path/to/OTA-pose \
    --train
```

### 2.2 PoseTrack2021

```bash
cd /path/to/OTA-pose

# Communication port (can be changed if occupied)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --use-env \
    --nproc_per_node=2 \
    --master_port=29511 \
    tools/run.py \
    --cfg configs/otapose/PE/posetrack21/no_supp_targ_5_frame.yaml \
    --root_dir /path/to/OTA-pose \
    --train
```

**Note:** The `--root_dir` argument specifies the project root directory and is used to resolve the relative paths defined in the configuration files.

## 3. Acknowledgements

We thank the [DCPose](https://github.com/Pose-Group/DCPose) project for providing supplementary files used in our data preparation, and the [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) project for providing pretrained weights.
