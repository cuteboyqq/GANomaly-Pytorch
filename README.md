# AutoEncoder-Pytorch

No Discriminator model 

# implement 

1. Encoder--Decoder--Encoder

2. loss function

3. Encoder/Decoder : Use paper network, conv--batchnorm--leakyrelu


![image](https://user-images.githubusercontent.com/58428559/187032363-003a6ef7-82b6-4829-a72f-000c9e4a1d86.png)

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
python train.py
```

# Test
```
python test.py
```
Example :
Train dataset : factory line only

dataset :factory line , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187033159-156e3b7d-35e9-4720-8c05-7420a7dda0eb.png)

dataset :factory noline , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187033196-c5d015a6-b71d-4bfd-a38a-cdae8e889455.png)

# Reference : 

https://arxiv.org/abs/1805.06725

