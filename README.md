## CV-SSL-MIS
Exploring CNN and ViT for Semi-Supervised Medical Image Segmentation


## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Efficientnet-Pytorch
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......


## Usage

1. Clone the repo:
```
git clone https://github.com/ziyangwang007/CV-SSL-MIS.git 
cd CV-SSL-MIS
```
2. Download the processed data and put the data in `../data/BraTS2019` or `../data/ACDC`.

```
TBC where to download
```

3. Train the model
```
cd code
python train_XXXXX.py
```

4. Test the model
```
python test_XXXXX.py
```



## References
This code is originally developed from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), and please consider citing the following works, if you use in your research/projects:

	@misc{ssl4mis2020,
	  title={{SSL4MIS}},
	  author={Luo, Xiangde},
	  howpublished={\url{https://github.com/HiLab-git/SSL4MIS}},
	  year={2020}}

	@misc{CVSSLMIS2022,
	  title={{CVSSLMIS}},
	  author={Luo, Xiangde},
	  howpublished={\url{https://github.com/ziyangwang007/CV-SSL-MIS}},
	  year={2022}}

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
	  booktitle={MICCAI Workshop on Resource-Efficient Medical Image Analysis},
	  pages={42--54},
	  year={2022},
	  organization={Springer}
	}

	@inproceedings{wang2022uncertainty,
	  title={An uncertainty-aware transformer for MRI cardiac semantic segmentation via mean teachers},
	  author={Wang, Ziyang and Zheng, Jian-Qing and Voiculescu, Irina},
	  booktitle={Annual Conference on Medical Image Understanding and Analysis},
	  pages={494--507},
	  year={2022},
	  organization={Springer}
	}

	TBC


## Acknowledgement

Part of the code is adapted from open-source codebase and original implementations of algorithms: [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [SegFormer](https://github.com/NVlabs/SegFormer), [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet), [segmentatic_segmentation.pytorch](https://github.com/qubvel/segmentation_models.pytorch).
