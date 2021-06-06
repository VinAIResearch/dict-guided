##### Table of Content

1. [Introduction](#dictionary-guided-scene-text-recognition)
1. [Dataset](#dataset)
1. [Getting Started](#getting-started)
	- [Requirements](#requirements)
	- [Usage Example](#usage)
1. [Training & Evaluation](#training-and-evaluation)
1. [Acknowledgement](#acknowledgement)

# Dictionary-guided Scene Text Recognition

- We propose a novel dictionary-guided sense text recognition approach that could be used to improve many state-of-the-art models.
- We also introduce a new benchmark dataset (namely, VinText) for Vietnamese scene text recognition.


| ![architecture.png](https://user-images.githubusercontent.com/32253603/117981172-ebd78580-b35e-11eb-84fe-b97c8d15d8bf.png) |
|:--:|
| *Comparison between the traditional approach and our proposed approach.*|

Details of the dataset construction, model architecture, and experimental results can be found in [our following paper](https://www3.cs.stonybrook.edu/~minhhoai/papers/vintext_CVPR21.pdf):

```
@inproceedings{m_Nguyen-etal-CVPR21,
      author = {Nguyen Nguyen and Thu Nguyen and Vinh Tran and Triet Tran and Thanh Ngo and Thien Nguyen and Minh Hoai},
      title = {Dictionary-guided Scene Text Recognition},
      year = {2021},
      booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    }
```
**Please CITE** our paper whenever our dataset or model implementation is used to help produce published results or incorporated into other software.

---

### Dataset

We introduce ✨ a new [VinText](https://drive.google.com/file/d/1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml/view?usp=sharing) dataset. 
> ***By downloading this dataset, USER agrees:***
> 
> * to use this dataset for research or educational purposes only
> * to not distribute or part of this dataset in any original or modified form.
> * and to [cite our paper](#dictionary-guided-scene-text-recognition) whenever this dataset are employed to help produce published results.

|    Name  						  | #imgs | #text instances						   | Examples 									|
|:-------------------------------:|:-----:|:-----------------------------------|:----------------------------------:|
|[VinText](https://drive.google.com/file/d/1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml/view?usp=sharing)| 2000  | About 56000 			   |![example.png](https://user-images.githubusercontent.com/32253603/120605880-c67afa80-c478-11eb-8a2a-039a1d316503.png)|

Detail about ✨ VinText dataset can be found in [our paper](https://www3.cs.stonybrook.edu/~minhhoai/papers/vintext_CVPR21.pdf).
Download this converted format dataset to fit with the model
- [Converted dataset](#Converted-dataset) - Converted format dataset to fit with model directly [Download here](https://drive.google.com/file/d/1AXl2iOTvLtMG8Lg2iU6qVta8VuWSXyns/view?usp=sharing)


### VinText
Extract data and copy folder to folder ```datasets/```

```
datasets
└───vietnamese
	└───test.json
		│train.json
		|train_images
		|test_images
└───evaluation
	└───gt_vietnamese.zip
```
---

### Getting Started

##### Requirements

- python=3.7
- torch==1.4.0
- detectron2==0.2

##### Installation

```sh
conda create -n dict-guided -y python=3.7
conda activate dict-guided
conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch
python -m pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 weighted-levenshtein editdistance

# Install Detectron2
python -m pip install detectron2==0.2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/torch1.4/index.html
```
### Check out the code and install: 
```sh
git clone https://github.com/nguyennm1024/dict-guided.git
cd dict-guided
python setup.py install
```

##### Download pre-trained model

- [vietnamese_trained_model](https://drive.google.com/file/d/15rJsQCO1ewJe-EInN-V5dSCftew4vLRz/view?usp=sharing).

##### Usage
| ![qualitative results.png](https://user-images.githubusercontent.com/32253603/120606555-836d5700-c479-11eb-9a37-09fa8cc129f3.png) |
|:--:|
| *Qualitative Results on VinText.*|


Prepare folders
```sh
mkdir sample_input
mkdir sample_output
```
Copy your images to ```sample_input/```. Output images would result in ```sample_output/```
```sh
python demo/demo.py --config-file configs/BAText/Vietnamese/attn_R_50.yaml --input sample_input/ --output sample_output/ --opts MODEL.WEIGHTS your_checkpoint.pth
```


### Training and Evaluation

```MODEL.WEIGHTS``` is a command line parameter that points to your checkpoint path

```checkpoint_name.pth``` is the name of the checkpoint that you want to use.

#### Training

We produce our results in VinText dataset by using checkpoint was provided in ABCNet repository as the pretrained. It was trained from Total Text dataset. Download the checkpoint: [tt_attn_R_50](https://cloudstor.aarnet.edu.au/plus/s/tYsnegjTs13MwwK/download)

```sh
python tools/train_net.py --config-file configs/BAText/Vietnamese/attn_R_50.yaml MODEL.WEIGHTS path_to_checkpoint/checkpoint_name.pth
```

Example:
```sh
python tools/train_net.py --config-file configs/BAText/Vietnamese/attn_R_50.yaml MODEL.WEIGHTS ./tt_attn_R_50.pth
```

#### Evaluation

```sh
python tools/train_net.py --eval-only --config-file configs/BAText/Vietnamese/attn_R_50.yaml MODEL.WEIGHTS path_to_checkpoint/checkpoint_name.pth
```
Example:
```sh
python tools/train_net.py --eval-only --config-file configs/BAText/Vietnamese/attn_R_50.yaml MODEL.WEIGHTS ./trained_model.pth
```
### Acknowledgement
This repository is built based-on [ABCNet](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BAText)
