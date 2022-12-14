## CV-SSL-MIS
Exploring CNN and ViT for Semi-Supervised Medical Image Segmentation


## Requirements
* [Pytorch]
* TensorBoardX
* Efficientnet-Pytorch
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......


## Usage

1. Clone the repo:
```
git clone https://github.com/ziyangwang007/CV-SSL-MIS.git 
cd CV-SSL-MIS
```
2. Download the processed data and put the data in `../data/BraTS2019` or `../data/ACDC`. In this project, we use ACDC for 2D purpose, and BraTS for 3D purpose. You can download the dataset with the list of labeled training, unlabeled training, validation, and testing slices as following:


ACDC from [Link](https://drive.google.com/file/d/1erKoNzknobgn7gZYEXylsJFYqq-gc6xQ/view?usp=share_link).

BraTS from [Link](https://drive.google.com/file/d/1erKoNzknobgn7gZYEXylsJFYqq-gc6xQ/view?usp=share_link).


3. Train the model
```
cd code
python train_XXX.py --root_path ../data/XXX --exp ACDC/XXX --model XXX -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 4 --labeled_num XXX
```

4. Test the model
```
python test_XXX.py -root_path ../data/XXX --exp ACDC/XXX -model XXX --num_classes 4 --labeled_num XXX
```

You can choose model, dataset, experiment name, iteration number, batch size and etc in your command line, or leave it with default option.

## References

Please consider citing the following works, if you use in your research/projects:

	@inproceedings{wang2022computationally,
	  title={Computationally-efficient vision transformer for medical image semantic segmentation via dual pseudo-label supervision},
	  author={Wang, Ziyang and Dong, Nanqing and Voiculescu, Irina},
	  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
	  pages={1961--1965},
	  year={2022},
	  organization={IEEE}
	}

	@inproceedings{wang2022triple,
	  title={Triple-view feature learning for medical image segmentation},
	  author={Wang, Ziyang and Voiculescu, Irina},
	  booktitle={Medical Image Computing and Computer Assisted Intervention Workshop (MICCAI-W)},
	  pages={42--54},
	  year={2022},
	  organization={Springer}
	}

	@inproceedings{wang2022uncertainty,
	  title={An uncertainty-aware transformer for MRI cardiac semantic segmentation via mean teachers},
	  author={Wang, Ziyang and Zheng, Jian-Qing and Voiculescu, Irina},
	  booktitle={Annual Conference on Medical Image Understanding and Analysis (MIUA)},
	  pages={494--507},
	  year={2022},
	  organization={Springer}
	}
<!-- 
	@inproceedings{wang2022adversarial,
	  title={Adversarial Vision Transformer for Medical Image Semantic Segmentation with Limited Annotations},
	  author={Wang, Ziyang and Zhao, Chengkuan and Ni, Zixuan},
	  booktitle={33rd British Machine Vision Conference (BMVC)},
	  year={2022}
	}

	@inproceedings{wang2022when,
	  title={When CNN Meet with ViT: Towards Semi-Supervised Learning for Multi-Class Medical Image Semantic Segmentation},
	  author={Wang, Ziyang and Li, Tianze and Zheng, Jian-Qing and Huang, Baoru},
	  booktitle={European Conference on Computer Vision Workshop (ECCV-W)},
	  year={2022},
	  organization={Springer}
	} -->

	Exigent Examiner and Mean Teacher: A Novel 3D CNN-based Semi-Supervised Learning Framework for Brain Tumor Segmentation (TBC)



## Acknowledgement

This code is mainly borrowed from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).

	@misc{ssl4mis2020,
	  title={{SSL4MIS}},
	  author={Luo, Xiangde},
	  howpublished={\url{https://github.com/HiLab-git/SSL4MIS}},
	  year={2020}}

Some of the other code is based on [SegFormer](https://github.com/NVlabs/SegFormer), [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet), [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch), [UAMT](https://github.com/yulequan/UA-MT), [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
