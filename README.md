# AutoEncoder-Pytorch

No Discriminator model 

implement :

Encoder--Decoder--Encoder

loss function

Encoder/Decoder : Use paper network, conv--batchnorm--leakyrelu

Reference : https://arxiv.org/abs/1805.06725

![image](https://user-images.githubusercontent.com/58428559/187032363-003a6ef7-82b6-4829-a72f-000c9e4a1d86.png)

Example :
Train dataset : factory line only

dataset :factory line , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187033159-156e3b7d-35e9-4720-8c05-7420a7dda0eb.png)

dataset :factory noline , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187033196-c5d015a6-b71d-4bfd-a38a-cdae8e889455.png)


UPDATE:
This repo is no longer maintained. GANomaly implementation has been added to anomalib, the largest public collection of ready-to-use deep learning anomaly detection algorithms and benchmark datasets.

GANomaly
This repository contains PyTorch implementation of the following paper: GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training [1]

1. Table of Contents
GANomaly
Table of Contents
Installation
Experiment
Training
Training on MNIST
Training on CIFAR10
Train on Custom Dataset
Citing GANomaly
Reference
2. Installation
First clone the repository
git clone https://github.com/samet-akcay/ganomaly.git
Create the virtual environment via conda
conda create -n ganomaly python=3.7
Activate the virtual environment.
conda activate ganomaly
Install the dependencies.
conda install -c intel mkl_fft
pip install --user --requirement requirements.txt
3. Experiment
To replicate the results in the paper for MNIST and CIFAR10 datasets, run the following commands:

# MNIST
sh experiments/run_mnist.sh

# CIFAR
sh experiments/run_cifar.sh # CIFAR10
4. Training
To list the arguments, run the following command:

python train.py -h
4.1. Training on MNIST
To train the model on MNIST dataset for a given anomaly class, run the following:

python train.py \
    --dataset mnist                         \
    --niter <number-of-epochs>              \
    --abnormal_class <0,1,2,3,4,5,6,7,8,9>  \
    --display                               # optional if you want to visualize     
4.2. Training on CIFAR10
To train the model on CIFAR10 dataset for a given anomaly class, run the following:

python train.py \
    --dataset cifar10                                                   \
    --niter <number-of-epochs>                                          \
    --abnormal_class                                                    \
        <plane, car, bird, cat, deer, dog, frog, horse, ship, truck>    \
    --display                       # optional if you want to visualize        
4.3. Train on Custom Dataset
To train the model on a custom dataset, the dataset should be copied into ./data directory, and should have the following directory & file structure:

Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

Then model training is the same as training MNIST or CIFAR10 datasets explained above.

python train.py                     \
    --dataset <name-of-the-data>    \
    --isize <image-size>            \
    --niter <number-of-epochs>      \
    --display                       # optional if you want to visualize
For more training options, run python train.py -h.

5. Citing GANomaly
If you use this repository or would like to refer the paper, please use the following BibTeX entry

@inproceedings{akcay2018ganomaly,
  title={Ganomaly: Semi-supervised anomaly detection via adversarial training},
  author={Akcay, Samet and Atapour-Abarghouei, Amir and Breckon, Toby P},
  booktitle={Asian Conference on Computer Vision},
  pages={622--637},
  year={2018},
  organization={Springer}
}
6. Reference
[1] Akcay S., Atapour-Abarghouei A., Breckon T.P. (2019) GANomaly: Semi-supervised Anomaly Detection via Adversarial Training. In: Jawahar C., Li H., Mori G., Schindler K. (eds) Computer Vision – ACCV 2018. ACCV 2018. Lecture Notes in Computer Science, vol 11363. Springer, Cham
