##### Table of Content

1. [Introduction](#cpm-color-pattern-makeup-transfer)
1. [Datasets](#datasets)
1. [Getting Started](#getting-started)
	- [Requirements](#requirements)
	- [Usage Example](#usage)
1. [Training & Evaluation](#training-and-evaluation)

# Dictionary-guided Scene Text Recognition

- Dictionary-guided is a novel approach that could be used to improve many state-of-the-art models.
- This repository is built based-on [ABCNet](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BAText).
- We also introduce a new Vietnamese scene text dataset (VinText) as a new benchmark for scene text spotting community in the future.


| ![architecture.png](https://user-images.githubusercontent.com/32253603/117981172-ebd78580-b35e-11eb-84fe-b97c8d15d8bf.png) |
|:--:|
| *Comparision of traditional approach and our proposal.*|

Details of the dataset construction, model architecture, and experimental results can be found in [our following paper](https://www3.cs.stonybrook.edu/~minhhoai/papers/vintext_CVPR21.pdf):

```
@inproceedings{m_Nguyen-etal-CVPR21,
      author = {Nguyen Nguyen and Thu Nguyen and Vinh Tran and Triet Tran and Thanh Ngo and Thien Nguyen and Minh Hoai},
      title = {Dictionary-guided Scene Text Recognition},
      year = {2021},
      booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    }
```
**Please CITE** our paper whenever our datasets or model implementation is used to help produce published results or incorporated into other software.

---

### Datasets

We introduce ✨ a new [VinText](https://drive.google.com/file/d/1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml/view?usp=sharing) dataset.
*Dataset Folder Structure can be found [here](https://github.com/VinAIResearch/dict-guided/blob/main/about-data.md).*
> ***By downloading these datasets, USER agrees:***
> 
> * to use these datasets for research or educational purposes only
> * to not distribute or part of these datasets in any original or modified form.
> * and to [cite our paper](#cpm-color-pattern-makeup-transfer) whenever these datasets are employed to help produce published results.

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

##### Download pre-trained models

- Download Vietnamese’s pre-trained models: [vietnamese_trained_model](https://drive.google.com/file/d/15rJsQCO1ewJe-EInN-V5dSCftew4vLRz/view?usp=sharing).

##### Usage
Prepare folders
```sh
mkdir sample_input
mkdir sample_output
```
Please copy your images to ```sample_input/```
```sh
python demo/demo.py --config-file configs/BAText/Vietnamese/attn_R_50.yaml --input sample_input/ --output sample_output/ --opts MODEL.WEIGHTS your_checkpoint.pth
```

Result image will be saved in `sample_output/`

### Training and Evaluation

#### Training

Fine-tune from checkpoint
```sh
python tools/train_net.py --config-file configs/BAText/Vietnamese/attn_R_50.yaml MODEL.WEIGHTS your_checkpoint.pth
```

#### Evaluation

```sh
python tools/train_net.py --eval-only --config-file configs/BAText/Vietnamese/attn_R_50.yaml MODEL.WEIGHTS your_checkpoint.pth
```
