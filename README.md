# AutoEncoder-Pytorch

No Discriminator model 

# implement 

1. Encoder--Decoder--Encoder

2. loss function

3. Encoder/Decoder : Use paper network, conv--batchnorm--leakyrelu


![image](https://user-images.githubusercontent.com/58428559/187032363-003a6ef7-82b6-4829-a72f-000c9e4a1d86.png)

# Train factory line dataset

```
\---2022-08-26
    \---f_384_2min
        +---crops
        |   +---blue_line
        |   +---gray_line
        |   \---Green_line

```
# Test noline (defect) dataset
```
\---2022-08-26
    \---f_384_2min  
        \---defeat
            +---noline
            \---others
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

