# Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Shifeng Zhang](http://www.cbsr.ia.ac.cn/users/sfzhang/), [Cheng Chi](https://chicheng123.github.io/), [Yongqiang Yao](https://github.com/yqyao), [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/), [Stan Z. Li](http://www.cbsr.ia.ac.cn/users/szli/).

## Introduction

In this work, we first point out that the essential difference between anchor-based and anchor-free detection is actually **how to define positive and negative training samples**. Then we propose an Adaptive Training Sample Selection (ATSS) to automatically select positive and negative samples according to statistical characteristics of object, which significantly improves the performance of anchor-based and anchor-free detectors and bridges the gap between them. Finally, we demonstrate that tiling multiple anchors per location on the image to detect objects is a thankless operation under current situations. Extensive experiments conducted on MS COCO support our aforementioned analysis and conclusions. With the newly introduced ATSS, we improve state-of-the-art detectors by a large margin to 50.7% AP without introducing any overhead. For more details, please refer to our [paper](https://arxiv.org/abs/1912.02424).

*Note: The lite version of our ATSS has been merged to the official code of [FCOS](https://github.com/tianzhi0549/FCOS) as the [center sampling](https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/modeling/rpn/fcos/loss.py#L166-L173) improvement, which improves its performance by ~0.8%. The full version of our ATSS can further improve its performance.*

## Installation
This ATSS implementation is based on [FCOS](https://github.com/tianzhi0549/FCOS) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and the installation is the same as them. Please check [INSTALL.md](INSTALL.md) for installation instructions.

## A quick demo
Once the installation is done, you can download *ATSS_R_50_FPN_1x.pth* from [Google](https://drive.google.com/open?id=1t8RLdQ6fsFXa0kzPIQ7541uZeQeMXP73) or [Baidu](https://pan.baidu.com/s/1bYXjWJE35kHLpQAIeWtZ0g) to run a quick demo.
    
    # assume that you are under the root directory of this project,
    # and you have activated your virtual environment if needed.
    python demo/atss_demo.py
    


## Inference
The inference command line on coco minival split:

    python tools/test_net.py \
        --config-file configs/atss/atss_R_50_FPN_1x.yaml \
        MODEL.WEIGHT ATSS_R_50_FPN_1x.pth \
        TEST.IMS_PER_BATCH 4    

Please note that:
1) If your model's name is different, please replace `ATSS_R_50_FPN_1x.pth` with your own.
2) If you enounter out-of-memory error, please try to reduce `TEST.IMS_PER_BATCH` to 1.
3) If you want to evaluate a different model, please change `--config-file` to its config file (in [configs/atss](configs/atss)) and `MODEL.WEIGHT` to its weights file.

## Models
For your convenience, we provide the following trained models. All models are trained with 16 images in a mini-batch and frozen batch normalization (i.e., consistent with models in [FCOS](https://github.com/tianzhi0549/FCOS) and [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)).*

Model | Multi-scale training | Testing time / im | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:|:---:
ATSS_R_50_FPN_1x | No | 44ms | 39.3 | 39.3 | [Google](https://drive.google.com/open?id=1t8RLdQ6fsFXa0kzPIQ7541uZeQeMXP73)/[Baidu](https://pan.baidu.com/s/1bYXjWJE35kHLpQAIeWtZ0g)
ATSS_dcnv2_R_50_FPN_1x | No | 54ms | 43.2 | 43.0 | [Google](https://drive.google.com/open?id=1_Zl6sVrNZbvawxtMdvNSE9wgURmkLLka)/[Baidu](https://pan.baidu.com/s/1baZJMCCy_waR0hhChgEQFA)
ATSS_R_101_FPN_2x | Yes | 57ms | 43.5 | 43.6 | [Google](https://drive.google.com/open?id=1jenAgiLLqome8nn5ghV7wmknfr1Xg_Dw)/[Baidu](https://pan.baidu.com/s/1hiAew46s877dpgAZ-AweLw)
ATSS_dcnv2_R_101_FPN_2x | Yes | 73ms | 46.1 | 46.3 | [Google](https://drive.google.com/open?id=17S-M6UILyS18s5RW1T6lWFi8nrKMhwd7)/[Baidu](https://pan.baidu.com/s/1eakRoQIqR-UmjWT4RM8vyQ)
ATSS_X_101_32x8d_FPN_2x | Yes | 110ms | 44.8 | 45.1 | [Google](https://drive.google.com/open?id=1jFTdsQD2KfR9Dh1NgX05_02wfQxlnmD3)/[Baidu](https://pan.baidu.com/s/1uO3ZLstI7tkVQBayjRy-6w)
ATSS_dcnv2_X_101_32x8d_FPN_2x | Yes | 143ms | 47.7 | 47.7 | [Google](https://drive.google.com/open?id=19E7vh7YCq0ZpvRIaswDMWGRmwcGK56Bz)/[Baidu](https://pan.baidu.com/s/1pOMZGb3UZb7u_lTqUk55Mw)
ATSS_X_101_64x4d_FPN_2x | Yes | 112ms | 45.5 | 45.6 | [Google](https://drive.google.com/open?id=1ECj7mQwZowiTsSwDXU5Q_Ab2tG-Byhsk)/[Baidu](https://pan.baidu.com/s/1LxNkz0To_mGWGRbtzA78bw)
ATSS_dcnv2_X_101_64x4d_FPN_2x | Yes | 144ms | 47.7 | 47.7 | [Google](https://drive.google.com/open?id=1Lmhtn71AgJC_6B5iqU8-PG_rYanKEr2k)/[Baidu](https://pan.baidu.com/s/1nzX-lUvZfnV--fj6OwsnmQ)

[1] *The testing time is taken from [FCOS](https://github.com/tianzhi0549/FCOS), because our method only redefines positive and negative training samples without incurring any additional overhead.* \
[2] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[3] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[4] *`dcnv2` denotes deformable convolutional networks v2. Note that for ResNet based models, we apply deformable convolutions from stage c3 to c5 in backbones. For ResNeXt based models, only stage c4 and c5 use deformable convolutions. All models use deformable convolutions in the last layer of detector towers.* \
[5] *The model `ATSS_dcnv2_X_101_64x4d_FPN_2x` with multi-scale testing achieves 50.7% in AP on COCO test-dev. Please use `TEST.BBOX_AUG.ENABLED True` to enable multi-scale testing.*

## Training

The following command line will train ATSS_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/atss/atss_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/atss_R_50_FPN_1x
        
Please note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/atss/atss_R_50_FPN_1x.yaml](configs/atss/atss_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train ATSS with other backbones, please change `--config-file`.

## Contributing to the project
Any pull requests or issues are welcome.

## Citations
Please cite our paper in your publications if it helps your research:
```
@inproceedings{zhang2020bridging,
  title     =  {Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection},
  author    =  {Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z.},
  booktitle =  {CVPR},
  year      =  {2020}
}
```
