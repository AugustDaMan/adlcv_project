# VPD

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unleashing-text-to-image-diffusion-models-for-1/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=unleashing-text-to-image-diffusion-models-for-1)


Created by [Wenliang Zhao](https://wl-zhao.github.io/)\*, [Yongming Rao](https://raoyongming.github.io/)\*,  [Zuyan Liu](https://scholar.google.com/citations?user=7npgHqAAAAAJ&hl=en)\*, [Benlin Liu](https://liubl1217.github.io), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)†

This repository contains PyTorch implementation for paper "Unleashing Text-to-Image Diffusion Models for Visual Perception" (ICCV 2023). 

VPD (<ins>**V**</ins>isual <ins>**P**</ins>erception with Pre-trained <ins>**D**</ins>iffusion Models) is a framework that leverages the high-level and low-level knowledge of a pre-trained text-to-image diffusion model to downstream visual perception tasks.

![intro](figs/intro.png)

[[Project Page]](https://vpd.ivg-research.xyz) [[arXiv]](https://arxiv.org/abs/2303.02153)


## Installation
Clone this repo, and run
```
git submodule init
git submodule update
```
Download the checkpoint of [stable-diffusion](https://github.com/runwayml/stable-diffusion) (we use `v1-5` by default) and put it in the `checkpoints` folder. Please also follow the instructions in [stable-diffusion](https://github.com/runwayml/stable-diffusion) to install the required packages.

## Semantic Segmentation with VPD
Equipped with a lightweight Semantic FPN and trained for 80K iterations on $512\times512$ crops, our VPD can achieve 54.6 mIoU on ADE20K.

Please check [segmentation.md](./segmentation/README.md) for detailed instructions.

## Referring Image Segmentation with VPD
VPD achieves 73.46, 63.93, and 63.12 oIoU on the validation sets of RefCOCO, RefCOCO+, and G-Ref, repectively.

| Dataset | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 | OIoU | Mean IoU
|:---:|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
RefCOCO | 85.52 | 83.02 | 78.45 | 68.53 | 36.31 | **73.46** | 75.67 
RefCOCO+ | 76.69 | 73.93 | 69.68 | 60.98 | 32.52 | **63.93** | 67.98 
RefCOCOg | 75.16 | 71.16 | 65.60 | 55.04 | 29.41 | **63.12** | 66.42 


Please check [refer.md](./refer/README.md) for detailed instructions on training and inference.

## Depth Estimation with VPD
VPD obtains 0.254 RMSE on NYUv2 depth estimation benchmark, establishing the new state-of-the-art.

|  | RMSE | d1 | d2 | d3 | REL  | log_10 |
|-------------------|-------|-------|--------|--------|--------|-------|
| **VPD** | 0.254 | 0.964 | 0.995 | 0.999 | 0.069 | 0.030 |

Please check [depth.md](./depth/README.md) for detailed instructions on training and inference.

## License
MIT License

## Acknowledgements
This code is based on [stable-diffusion](https://github.com/CompVis/stable-diffusion), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [LAVT](https://github.com/yz93/LAVT-RIS), and [MIM-Depth-Estimation](https://github.com/SwinTransformer/MIM-Depth-Estimation).

## Citation
If you find our work useful in your research, please consider citing:
```
@article{zhao2023unleashing,
  title={Unleashing Text-to-Image Diffusion Models for Visual Perception},
  author={Zhao, Wenliang and Rao, Yongming and Liu, Zuyan and Liu, Benlin and Zhou, Jie and Lu, Jiwen},
  journal={ICCV},
  year={2023}
}
```
