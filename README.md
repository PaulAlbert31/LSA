# Official code for the PLS-LSA algorithm, accepted at ECCV 2024: An accurate detection is not all you need to combat label noise in web-noisy datasets.

[![ECCV 2024](https://img.shields.io/badge/ECCV-2024-blue)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06511.pdf)
[![Paper](https://img.shields.io/badge/Paper-PDF-orange)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06511.pdf)
[![Poster](https://img.shields.io/badge/Poster-PDF-yellow)](https://eccv.ecva.net/media/PosterPDFs/ECCV%202024/2673.png?t=1726103500.6770017)


## Overview

This repository provides the official codebase for our ECCV 2024 paper: "An accurate detection is not all you need to combat label noise in web-noisy datasets."


## Architecture

![PLS-LSA Architecture](https://github.com/PaulAlbert31/LSA/blob/main/images/6511_thumb.png)


## Getting Started

### Requirements

All dependencies are listed in the [lsa.yml](https://github.com/PaulAlbert31/LSA/blob/main/lsa.yml) file.


### Installation

Create a Conda environment using:
```sh
conda env create -f lsa.yml
conda activate lsa
```

We use LightningLite's Fabric [fabric](https://github.com/Lightning-AI/pytorch-lightning?tab=readme-ov-file#lightning-fabric-expert-control) to enable multi-gpu support, although it is not currently implemented.

## Datasets

### Downloads
This repository supports the following noisy datasets:

* **Webvision**: Download the dataset from [Webvision 2017](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html) and follow the instructions. For faster training, we use the first 50 classes (mini-Webvision).
* **Controlled Noisy Web Label (CNWL)**: Download the dataset from the [official webpage](https://google.github.io/controlled-noisy-web-labels/index.html) or using TFrecords from [FaMUS repository](https://github.com/youjiangxu/FaMUS?tab=readme-ov-file#dataset).
* **Webly-fg**: Download the dataset from the official [repository](https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset).
* **ImageNet2012**: Download the test set from [ImageNet](https://image-net.org/download.php) for evaluation.

### Dataset Paths
Update the [mypath.py](https://github.com/PaulAlbert31/LSA/blob/main/lsa.yml) file with the paths to the downloaded datasets.

## Training

### Contrastive Pre-training
Pretrain using unsupervised algorithms (SimCLR) from the [solo-learn](https://github.com/vturrisi/solo-learn) codebase.

* Pre-trained weights for CNWL and Webvision experiments are available at [google drive](https://drive.google.com/drive/folders/1WGupEKQUTHBH0-mc4LJqgDC1c_ilk8_c?usp=sharing)
* Specify the path to pre-trained weights using the `--pretrained` argument.


### PLS-LSA and PLS-LSA+
Run experiments using the [train.sh](https://github.com/PaulAlbert31/LSA/blob/main/train.sh) file, which includes examples for:
	- PLS-LSA and PLS-LSA+ on CNWL, Webvision, and Webly-fg datasets
	- PLS-LSA and PLA-LSA+ for ViTs pre-trained using CLIP


## Getting Started
1. Download and prepare the datasets.
2. Update [mypath.py](https://github.com/PaulAlbert31/LSA/blob/main/mypath.py) with dataset paths.
3. Pretrain using contrastive learning (optional).
4. Run experiments using [train.sh](https://github.com/PaulAlbert31/LSA/blob/main/train.sh).

## Results

#### Performance on Benchmark Datasets

| Dataset | Result |
| --- | --- |
| CNWL | ![CNWL](https://github.com/PaulAlbert31/LSA/blob/main/images/CNWL.png) |
| Webvision | ![Webvision](https://github.com/PaulAlbert31/LSA/blob/main/images/Webvision.png) |
| Webly-fg | ![Webly-fg](https://github.com/PaulAlbert31/LSA/blob/main/images/Webly-fg.png) |
| CNWL (CLIP) | ![CNWL CLIP](https://github.com/PaulAlbert31/LSA/blob/main/images/CNWL_CLIP.png) |


### Reproduction Note

Please note that results reproduced using this codebase may slightly differ due to code cleanup and restructuring.


## Citation

### Citing Our Work

If you find our work useful for your research, please cite our paper:


```bibtex
@inproceedings{2024_ECCV_LSA,
  title={An accurate detection is not all you need to combat label noise in web-noisy datasets},
  author={Albert, Paul and Valmadre, Jack and Arazo, Eric and Krishna, Tarun and O'Connor, Noel E and McGuinness, Kevin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```