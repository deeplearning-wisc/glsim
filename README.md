# glsim

Source code for the NeurIPS 2025 paper: "[GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Embedding Similarity](https://arxiv.org/abs/2508.19972)" by Seongheon Park and Sharon Li.


## Setup and Installation

### 1. Environment Setup


```bash
conda create -n glsim python=3.9
conda activate glsim
```

### 2. Install Dependencies


```bash
pip install torch transformers Pillow tqdm scikit-learn numpy matplotlib nltk pattern
```

### 3. Download MSCOCO Dataset

The evaluation is performed on the MSCOCO dataset. You need to download the images and annotations.

1.  Download the 2014 validation images from the [COCO website](https://cocodataset.org/#download).
2.  Download the 2014 train/val annotations.

You will need to update the paths in `evaluate.py` to point to your local COCO dataset directory and annotation file.

-   `MSCOCO_DATASET_PATH`: Path to the directory containing COCO validation images (e.g., `val2014/`).
-   `COCO_ANNOTATION_PATH`: Path to the file containing COCO ground truth `coco_ground_truth.json`.

### 4. Generate CHAIR Cache


```bash
python util/chair.py --cache chair.pkl
```


## Usage

To run the evaluation, use the `evaluate.py` script.

```bash
python evaluate.py --lvlm llava-1.5-7b-hf 
```


## Citation


```bibtex
@inproceedings{
park2025glsim,
title={{GLS}im: Detecting Object Hallucinations in {LVLM}s via Global-Local Similarity},
author={Seongheon Park and Sharon Li},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=ZO8LyCizx9}
}
```
