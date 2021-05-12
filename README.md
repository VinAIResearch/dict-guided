<p align="center">	
<img width="350" height="200" alt="logo" src="https://user-images.githubusercontent.com/32253603/117951605-85db0600-b33e-11eb-8cc6-f659205e2055.png">
<img width="350" height="200" alt="logo" src="https://user-images.githubusercontent.com/32253603/117954945-b7090580-b341-11eb-9887-421bd618dde5.jpg">
</p>


# Dictionary-guided Scene Text Recognition

<p align="center">	
<img width="700" alt="logo" src="https://user-images.githubusercontent.com/32253603/117981172-ebd78580-b35e-11eb-84fe-b97c8d15d8bf.png">
</p>

Details of our model architecture and experimental results can be found in our [following paper](http://arxiv.org/abs/2101.01476):


## Installation

```sh
conda create -n dict-guided -y python=3.7
conda activate dict-guided
conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch
python -m pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 weighted-levenshtein

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

### Data download and preparation

[Download VinText dataset (original format)](https://drive.google.com/file/d/1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml/view?usp=sharing)

[Download converted VinText dataset](https://drive.google.com/file/d/1AXl2iOTvLtMG8Lg2iU6qVta8VuWSXyns/view?usp=sharing)

Extract data and copy folder to folder ```datasets/```

```
datasets/
  vietnamese/
    test.json
    train.json
    train_images/
    test_images/
  evaluation/
    gt_vietnamese.zip
```


## Usage example: Command lines

### Training

Fine-tune from checkpoint
```sh
python tools/train_net.py --config-file configs/BAText/Vietnamese/attn_R_50.yaml MODEL.WEIGHTS your_checkpoint.pth
```

### Evaluation

1. Evaluation
```sh
python tools/train_net.py --eval-only --config-file configs/BAText/Vietnamese/attn_R_50.yaml MODEL.WEIGHTS your_checkpoint.pth
```

2. Test and visualize your own images

Please copy your images to ```sample_input/```
```sh
python demo/demo.py --config-file configs/BAText/Vietnamese/attn_R_50.yaml --input sample_input/ --output sample_output/ --opts MODEL.WEIGHTS your_checkpoint.pth
```

#### The pre-trained model for Vietnamese is available [HERE](https://drive.google.com/file/d/15rJsQCO1ewJe-EInN-V5dSCftew4vLRz/view?usp=sharing)!

**Please CITE** our paper when model or data is used to help produce published results or incorporated into other software.

Dictionary-guided Scene Text Recognition. \
Nguyen Nguyen, Thu Nguyen, Vinh Tran, Triet Tran, Thanh Ngo, Thien Nguyen, Minh Hoai. \
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

    @inproceedings{m_Nguyen-etal-CVPR21,
      author = {Nguyen Nguyen and Thu Nguyen and Vinh Tran and Triet Tran and Thanh Ngo and Thien Nguyen and Minh Hoai},
      title = {Dictionary-guided Scene Text Recognition},
      year = {2021},
      booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    }
