# GANomaly-Pytorch
[(Back to top)](#table-of-contents)

Generator +  Discriminator model 


# Table of contents

<!-- After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README. -->

- [GANomaly-Pytorch](#GANomaly-Pytorch)
- [implement](#implement)
- [Train-on-custom-dataset](#Train-on-custom-dataset)
- [Train](#Train)
- [Test](#Test)
- [Lose-value-distribution](#Lose-value-distribution)
- [Reference](#Reference)
   

# implement 
[(Back to top)](#table-of-contents)

1. Encoder--Decoder--Encoder

2. loss function

3. Encoder/Decoder : Use paper network, conv--batchnorm--leakyrelu

4. Discriminator (2022/08/30 updated)

![image](https://user-images.githubusercontent.com/58428559/187453543-ea807f75-46ba-443f-ae89-a19bb8151f0d.png)



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
Example :
Train dataset : factory line only

dataset :factory line , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187036135-46cd0915-b695-48a8-b377-0859e57fb1da.png)


dataset :factory noline , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/187036162-52b6fb52-cc6b-44b6-99e5-d532332e9c9a.png)

# Lose-value-distribution
[(Back to top)](#table-of-contents)

Blue : normal dataset

Orange : abnormal dataset

![image](https://user-images.githubusercontent.com/58428559/187057006-1564dd37-aa9d-4261-9240-f2507156361f.png)




# Reference 
[(Back to top)](#table-of-contents)

https://arxiv.org/abs/1805.06725

