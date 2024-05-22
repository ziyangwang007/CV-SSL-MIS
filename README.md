## CV-SSL-MIS
Exploring CNN and ViT for Semi-Supervised Medical Image Segmentation


## Requirements
* Pytorch, MONAI 
* Some basic python packages: Torchio, Numpy, Scikit-image, SimpleITK, Scipy, Medpy, nibabel, tqdm ......

## News and To-do List
- [x] Contrastive Learning
- [x] Various Segmentation Backbone Networks 3D UNETR, 3D SwinUNETR, 3D UNet, nnUNet ... 
- [x] 3D UNETR
- [x] 3D SwinUNETR
- [x] 2D SwinUNet
- [x] ROMISE12 Prostate dataset
- [ ] 2D U-Mamba
- [ ] 3D SegMamba
- [ ] Totalsegmentor dataset

## Usage

1. Clone the repo:
```
git clone https://github.com/ziyangwang007/CV-SSL-MIS.git 
cd CV-SSL-MIS
```
2. Download the pre-processed data and put the data in `../data/BraTS2019` or `../data/ACDC` or `../data/Prostate` or `../data/TotalSegmentator`. In this project, we use ACDC, TotalSegmentator for 2D purpose, and BraTS for 3D purpose. You can download the dataset with the list of labeled training, unlabeled training, validation, and testing slices as following:


ACDC from [Google Drive Link](https://drive.google.com/file/d/1F3JzBSIURtFJkfcExBcT6Hu7Ar5_f8uv/view?usp=sharing), or [Baidu Netdisk Link](https://pan.baidu.com/s/1LS6VHujD8kvuQikbydOibQ) with passcode: 'kafc'.

BraTS from [Google Drive Link](https://drive.google.com/file/d/1erKoNzknobgn7gZYEXylsJFYqq-gc6xQ/view?usp=share_link), or [Baidu Netdisk Link](https://pan.baidu.com/s/1Z1pSRIfx_41JG3o1KwS27A) with passcode: 'kbj3'.

Prostate from [Google Drive Link](https://drive.google.com/file/d/1jboocJQq3VP0CowK0xrnPV5aqKQhqUaW/view?usp=sharing).

TotalSegmentator from [zenodo](https://zenodo.org/record/6802613), [Google Drive Link](https://drive.google.com/file/d/1JDtMissERwDw2ej-Cd_ABAslfNjy79Fc/view?usp=sharing) or [Baidu Netdisk Link](https://pan.baidu.com/s/1LrScKvcAXG6h3qsafeaKPw) with passcode: 'm1d8'.


3. Train the model

```
cd code
```

You can choose model(unet/vnet/pnet/unetr...) by `'--model'`, dataset(acdc/brats) by `'--root_path'`, ratio of labeled/unlabel training set(10%, 20%, 30%, 50%) by `'--labeled_num'`, experiment name(the path of saving your model weights and inference) by `'--exp'`, iteration number, batch size, multi-class classification and etc in your command line, or leave it with default option.


Fully Supervised - CNN (2D UNet) -> [Paper Link](https://arxiv.org/pdf/1505.04597.pdf)
```
python train_fully_supervised_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_fully_supervised_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Fully Supervised - CNN (3D UNet) -> [Paper Link](https://arxiv.org/abs/1606.06650)
```
python train_fully_supervised_3D.py --root_path ../data/BraTS2019 --exp BraTS/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Fully Supervised - ViT (2D SwinUNet) -> [Paper Link](https://arxiv.org/abs/2105.05537)
```
python train_fully_supervised_2D_ViT.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_fully_supervised_2D_ViT.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Fully Supervised - ViT (3D UNETR) -> [Paper Link](https://arxiv.org/abs/2103.10504)
```
python train_fully_supervised_3D_ViT.py --root_path ../data/BraTS2019 --exp BraTS/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Mean Teacher - CNN -> [Paper Link](https://arxiv.org/pdf/1703.01780.pdf)
```
python train_mean_teacher_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_mean_teacher_3D.py --root_path ../data/BraTS2019 --exp BraTS/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Mean Teacher - ViT -> [Paper Link](https://link.springer.com/chapter/10.1007/978-3-031-12053-4_37)
```
python train_mean_teacher_ViT.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_mean_teacher_ViT.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```


Uncertainty-Aware Mean Teacher - CNN -> [Paper Link](https://arxiv.org/pdf/1907.07034.pdf)
```
python train_uncertainty_aware_mean_teacher_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_uncertainty_aware_mean_teacher_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX

python train_uncertainty_aware_mean_teacher_3D.py --root_path ../data/BraTS2019 --exp BraTS/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Uncertainty-Aware Mean Teacher - ViT  -> [Paper Link](https://link.springer.com/chapter/10.1007/978-3-031-12053-4_37)
```
python train_uncertainty_aware_mean_teacher_ViT_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_uncertainty_aware_mean_teacher_ViT_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Adversarial - CNN -> [Paper Link](https://arxiv.org/pdf/1802.07934.pdf)
```
python train_adversarial_network_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_adversarial_network_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX

python train_adversarial_network_3D.py --root_path ../data/BraTS2019 --exp BraTS/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Adversarial - ViT 
```
python train_adversarial_network_2D_ViT.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_adversarial_network_2D_ViT.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Cross Pseudo Supervision CNN -> [Paper Link](https://arxiv.org/pdf/2106.01226.pdf)
```
python train_cross_pseudo_supervision_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_cross_pseudo_supervision_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX

python train_cross_pseudo_supervision_3D.py --root_path ../data/BraTS2019 --exp BraTS/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Cross Pseudo Supervision - ViT CNN  -> [Paper Link](https://arxiv.org/pdf/2112.04894.pdf)
```
python train_cross_teaching_between_cnn_transformer_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_cross_teaching_between_cnn_transformer_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Cross Pseudo Supervision - ViT  -> [Paper Link](https://ieeexplore.ieee.org/abstract/document/9897482/)
```
python train_cross_pseudo_supervision_2D_ViT.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_cross_pseudo_supervision_2D_ViT.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Contrastive Learning - Cross Pseudo Supervision - CNN ViT
```
python train_Contrastive_Cross_CNN_ViT_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_Contrastive_Cross_CNN_ViT_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Contrastive Learning - Cross Pseudo Supervision - CNN -> [Paper Link](https://ieeexplore.ieee.org/abstract/document/10098633/) 
```
python train_Contrastive_Cross_CNN_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_Contrastive_Cross_CNN_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Fixmatch - CNN -> [Paper Link](https://arxiv.org/pdf/2001.07685.pdf)
```
python train_Fixmatch_CNN_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_Fixmatch_CNN_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX
```

Contrastive Learning - Fixmatch - Mean Teacher - ViT -> [Paper Link](https://openaccess.thecvf.com/content/ICCV2023W/NIVT/papers/Wang_Dual-Contrastive_Dual-Consistency_Dual-Transformer_A_Semi-Supervised_Approach_to_Medical_Image_Segmentation_ICCVW_2023_paper.pdf)
```
python train_Contrastive_Consistency_ViT_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_Contrastive_Consistency_ViT_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Adversarial Consistency - ViT  -> [Paper Link](https://bmvc2022.mpi-inf.mpg.de/1002.pdf)
```
python train_adversarial_consistency_ViT_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_adversarial_consistency_ViT_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Co-Training - CNN -> [Paper Link](https://arxiv.org/pdf/1803.05984.pdf)
```
python train_deep_co_training_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_deep_co_training_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Co-Training - ViT 
```
python train_deep_co_training_2D_ViT.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_deep_co_training_2D_ViT.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

MixUp - CNN -> [Paper Link](https://arxiv.org/abs/1903.03825)
```
python train_interpolation_consistency_training_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_interpolation_consistency_training_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX

python train_interpolation_consistency_training_3D.py --root_path ../data/BraTS2019 --exp BraTS/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

MixUp - ViT 
```
python train_interpolation_consistency_training_2D_ViT.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_interpolation_consistency_training_2D_ViT.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Semi CNN-ViT  -> [Paper Link](https://arxiv.org/pdf/2208.06449.pdf)
```
python train_cnn_meet_vit_2D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_cnn_meet_vit_2D.py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Triple-View Segmentation CNN -> [Paper Link](https://arxiv.org/pdf/2208.06303.pdf)
```
python train_tripleview_2D(demo).py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX

python train_tripleview_2D(demo).py --root_path ../data/Prostate --exp Prostate/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

Examiner-Student-Teacher CNN -> [Paper Link](https://link.springer.com/chapter/10.1007/978-3-031-44917-8_17)
```
python train_exam_student_teacher_3D.py --root_path ../data/ACDC --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 2 --labeled_num XXX
```

4. Test the model
```
python test_2D_fully.py -root_path ../data/XXX --exp ACDC/XXX -model XXX --num_classes 4 --labeled_num XXX

python test_3D.py -root_path ../data/XXX --exp ACDC/XXX -model XXX --num_classes 4 --labeled_num XXX

python test_CNNVIT.py -root_path ../data/XXX --exp ACDC/XXX -model XXX --num_classes 4 --labeled_num XXX
```
Check trained model and inference
```
cd model
```


## Acknowledgement

This code is mainly based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [MONAI](https://github.com/Project-MONAI/research-contributions/tree/main).

Some of the other code is from [SegFormer](https://github.com/NVlabs/SegFormer), [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet), [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch), [UAMT](https://github.com/yulequan/UA-MT), [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
