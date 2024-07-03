# GRADE
GRADE
## Graph Attentive Dual Ensemble learning for Unsupervised Domain Adaptation on point clouds
### Requirements

    * Pytorch 
    * Python 3.8

### Demo PointDA

Please enter the main folder and model

    python ../trainer_mt_2_3_2.py --exp_name M_Sh_mt --out_path ../experiments_mt_seed50 --dataroot ../data --src_dataset modelnet --trgt_dataset shapenet --DefRec_on_src False --apply_PCM False --DefRec_weight 0.2 --lr 1e-3 --wd 5e-5 --pre_model="../experiments_pre/M_pre_lr-3/model.pt" --pre_model2="../experiments_pre/M_pre_lr-3_s50/model.pt" --batch_size 32 --seed 50

## Demo PointSegDA

python ../trainer_mt_o13_2.py --exp_name _a_f --out_path ../experiments_o13_2_tr --dataroot ../data/PointSegDAdataset --src_dataset adobe --trgt_dataset faust --model 'dgcnn_trs' --DefRec_weight 0.05 --lr 1e-4 --wd 5e-6 --pre_model="../experiments_tr_pre/a_pre/model.pt" --pre_model2="../experiments_tr_pre_s50/a_pre/model.pt" 


### Our pre-trained 
python ../trainer_pre.py --exp_name M_pre --out_path ../experiments_pre_f --dataroot ../data --src_dataset modelnet --DefRec_weight 0.2 --lr 1e-3 --wd 5e-5

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
}
### Reference
This project is based on https://github.com/IdanAchituve/DefRec_and_PCM

This project is based on 
