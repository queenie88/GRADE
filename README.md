# GRADE
GRADE
## Graph Attentive Dual Ensemble learning for Unsupervised Domain Adaptation on point clouds
### Requirements

    * Pytorch 
    * Python 3.8

### Demo PointDA

Please enter the main folder, and model

    

### Our pretrained models




### Citing this repository

If you find our work helpful in your research, please kindly cite our paper:

    Li Q, Yan C, Hao Q, et al. Graph Attentive Dual Ensemble learning for Unsupervised Domain Adaptation on point clouds[J]. Pattern Recognition, 2024: 110690.
   
 bib:
   
@article{LI2024110690,
title = {Graph Attentive Dual Ensemble learning for Unsupervised Domain Adaptation on point clouds},
journal = {Pattern Recognition},
volume = {155},
pages = {110690},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.110690},
url = {https://www.sciencedirect.com/science/article/pii/S0031320324004412},
author = {Qing Li and Chuan Yan and Qi Hao and Xiaojiang Peng and Li Liu},
keywords = {Point clouds, Unsupervised Domain Adaptation, Dual ensemble learning, Graph attentive module, Pseudo labels},
abstract = {Due to the annotation difficulty of point clouds, Unsupervised Domain Adaptation (UDA) is a promising direction to address unlabeled point cloud classification and segmentation. Recent works show that adding a self-supervised learning branch for target domain training consistently boosts UDA point cloud tasks. However, most of these works simply resort to geometric deformation, which ignores semantic information and is hard to bridge the domain gap. In this paper, we propose a novel self-learning strategy for UDA on point clouds, termed as Graph Attentive Dual Ensemble learning (GRADE), which delivers semantic information directly. Specifically, with a pre-training process on the source domain, GRADE further builds dual collaborative training branches on the target domain, where each of them constructs a temporal average teacher model and distills its pseudo labels to the other branch. To achieve faithful labels from each teacher model, we improve the popular DGCNN architecture by introducing a dynamic graph attentive module to mine the relation between local neighborhood points. We conduct extensive experiments on several UDA point cloud benchmarks, and the results demonstrate that our GRADE method outperforms the state-of-the-art methods on both classification and segmentation tasks with clear margins.}
}
### Reference

This project is based on https://github.com/Megvii-Nanjing/ML_GCN
