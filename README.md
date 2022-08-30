# GANomaly-Pytorch

No Discriminator model 

# implement 

1. Encoder--Decoder--Encoder

2. loss function

3. Encoder/Decoder : Use paper network, conv--batchnorm--leakyrelu

4. Discriminator (2022/08/30 updated)

![image](https://user-images.githubusercontent.com/58428559/187036065-f1b7f624-bb0d-4d96-b3c9-6e6d8f74c98a.png)



# Train on custom dataset

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
```
python train.py --img-dir "[train dataset dir]" --batch-size 64 --img-size 32 --epoch 20
```

# Test
```
python test.py --nomal-dir "[test normal dataset dir]" --abnormal-dir "[test abnormal dataset dir]" --view-img --img-size 32
```
Example :
Train dataset : factory line only

dataset :factory line , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187036135-46cd0915-b695-48a8-b377-0859e57fb1da.png)


dataset :factory noline , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187036162-52b6fb52-cc6b-44b6-99e5-d532332e9c9a.png)

# Lose value distribution

Blue : normal dataset

Orange : abnormal dataset

![image](https://user-images.githubusercontent.com/58428559/187057006-1564dd37-aa9d-4261-9240-f2507156361f.png)




# Reference : 

https://arxiv.org/abs/1805.06725

