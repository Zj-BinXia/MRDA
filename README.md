# MRDA
This project is the official implementation of 'Meta-Learning based Degradation Representation for Blind Super-Resolution', TIP2023

# KDSR-classic


This project is the official implementation of 'Meta-Learning based Degradation Representation for Blind Super-Resolution', TIP2023
> **Meta-Learning based Degradation Representation for Blind Super-Resolution [[Paper](https://arxiv.org/pdf/2207.13963.pdf)] [[Project](https://github.com/Zj-BinXia/MRDA)]**

This is code for MRDA (for classic degradation model, ie y=kx+n)

<p align="center">
  <img src="figs/process.jpg" width="80%">
</p>

---

##  Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.10](https://pytorch.org/)

## Dataset Preparation

We use DF2K, which combines [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 images) and [Flickr2K](https://github.com/LimBee/NTIRE2017) (2650 images).

---

## Training (4 V100 GPUs)

### Isotropic Gaussian Kernels

1. We train KDSRT-M ( using L1 loss)

```bash
sh main_iso_KDSRsMx4_stage3.sh 
```

2. we train KDSRS-M (using L1 loss and KD loss). **It is notable that modify the ''pre_train_TA'' and ''pre_train_ST'' of main_iso_KDSRsMx4_stage4.sh  to the path of trained KDSRT-M checkpoint.** Then, we run

```bash
sh main_iso_KDSRsMx4_stage4.sh 
```

### Anisotropic Gaussian Kernels plus noise

1. We train KDSRT ( using L1 loss)

```bash
sh main_anisonoise_KDSRsMx4_stage3.sh
```

2. we train KDSRS (using L1 loss and KD loss). **It is notable that modify the ''pre_train_TA'' and ''pre_train_ST'' of main_anisonoise_KDSRsMx4_stage4.sh  to the path of trained KDSRT checkpoint.** Then, we run

```bash
sh main_anisonoise_KDSRsMx4_stage4.sh
```

---

## :european_castle: Model Zoo

Please download checkpoints from [Google Drive](https://drive.google.com/drive/folders/113NBvfcrCedvend96KqDiRYVy3N8yprl).

---

## Testing

### Isotropic Gaussian Kernels

Test KDSRsM
```bash
sh test_iso_KDSRsMx4.sh
```
Test KDSRsL

```bash
sh test_iso_KDSRsLx4.sh
```
### Anisotropic Gaussian Kernels plus noise

```bash
sh test_anisonoise_KDSRsMx4.sh
```

---

## Results
<p align="center">
  <img src="images/iso-quan.jpg" width="90%">
</p>

<p align="center">
  <img src="images/aniso-quan.jpg" width="90%">
</p>


---

## BibTeX

@article{xia2022meta,
  title={Meta-learning based degradation representation for blind super-resolution},
  author={Xia, Bin and Tian, Yapeng and Zhang, Yulun and Hang, Yucheng and Yang, Wenming and Liao, Qingmin},
  journal={IEEE Transactions on Image Processing},
  year={2023}
}

