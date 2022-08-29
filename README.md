# AutoEncoder-Pytorch
[(Back to Table)](#table-of-contents)

No Discriminator model 


#  1. Table of Contents
- [AutoEncoder-Pytorch](#AutoEncoder-Pytorch)
    - [implement](#implement)
    - [Train on custom dataset](#Train-on-custom-dataset)
    - [Train](#Train)
    - [Test](#Test)
        - [Example](#Example)
        - [Lose value distribution](#Lose-value-distribution)
    - [Reference](#Reference)


## implement 
[(Back to top)](#table-of-contents)

1. Encoder--Decoder--Encoder

2. loss function

3. Encoder/Decoder : Use paper network, conv--batchnorm--leakyrelu

![image](https://user-images.githubusercontent.com/58428559/187036065-f1b7f624-bb0d-4d96-b3c9-6e6d8f74c98a.png)



# Train-on-custom-dataset
[(Back to top)](#table-of-contents)

```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png


```

# Train
[(Back to top)](#table-of-contents)

```
python train.py --img-dir "[train dataset dir]" --batch-size 64 --img-size 32 --epoch 20
```

# Test
[(Back to top)](#table-of-contents)

```
python test.py --nomal-dir "[test normal dataset dir]" --abnormal-dir "[test abnormal dataset dir]" --view-img --img-size 32
```
## Example
[(Back to top)](#table-of-contents)


## Train factory dataset
[(Back to top)](#table-of-contents)

factory line only, batch_size=64, img_size=64, nz=400, epoch=30

dataset :factory line , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187036135-46cd0915-b695-48a8-b377-0859e57fb1da.png)


dataset :factory noline , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187036162-52b6fb52-cc6b-44b6-99e5-d532332e9c9a.png)

# Lose-value-distribution
[(Back to top)](#table-of-contents)

Blue : normal dataset

Orange : abnormal dataset

![image](https://user-images.githubusercontent.com/58428559/187195639-ae90b89e-3f24-4718-9191-228ab83580d5.png)



# Reference
[(Back to top)](#table-of-contents)

https://arxiv.org/abs/1805.06725

