# TIG-UDA

## <p align="center">TIG-UDA: Generative unsupervised domain adaptation with transformer-embedded invariance for cross-modality medical image segmentation

<div align="center">

[Paper Link](https://www.sciencedirect.com/science/article/abs/pii/S1746809425002332)

</div>

  This repository provides the official implementation of the paper (Accepted by *Biomedical Signal Processing and Control*):  

## 🧠 Overview

<div align="center">
  <img src="/Fig/Overview.png" alt="TIG-UDA Overview" width="600"/> <!-- /Fig/Overview.png -->
</div>

*Abstract*: Unsupervised domain adaptation (UDA) in medical image segmentation aims to transfer knowledge from a labeled source domain to an unlabeled target domain, especially when there are significant differences in data distribution across multi-modal medical images. Traditional UDA methods typically involve image translation and segmentation modules. However, during image translation, the anatomical structure of the generated images may vary, resulting in a mismatch of source domain labels and impacting subsequent segmentation. In addition, during image segmentation, although the Transformer architecture is used in UDA tasks due to its superior global context capture ability, it may not effectively facilitate knowledge transfer in UDA tasks due to lacking the adaptability of the self-attention mechanism in Transformers. To address these issues, we propose a generative UDA network with invariance mining, named TIG-UDA, for cross-modality multi-organ medical image segmentation, which includes an image style translation network (ISTN) and an invariance adaptation segmentation network (IASN). In ISTN, we not only introduce a structure preservation mechanism to guide image generation to achieve anatomical structure consistency, but also align the latent semantic features of source and target domain images to enhance the quality of the generated images. In IASN, we propose an invariance adaptation module that can extract the invariability weights of learned features in the attention mechanism of Transformer to compensate for the differences between source and target domains. Experimental results on two public cross-modality datasets (MS-CMR dataset and Abdomen dataset) show the promising segmentation performance of TIG-UDA compared with other state-of-the-art UDA methods.


### Invariance Adaptation Module
<div align="center">
  <img src="/Fig/IAM.png" alt="IAM" width="600"/> <!-- /Fig/Overview.png -->
</div>

## 🛠️ Environment
This code is implemented in **PyTorch**. Below are the key environment configurations:
- Python 3.9
- You can install dependencies with:
```
pip install -r requirements.txt
```
Pretrained ViT
Download the following models and put them in /ISTN/checkpoints:
- [ViT-B_16](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.49067683.-40935391.1637977007) (ImageNet-21K)

## 🚀 Training Pipeline
This repository is divided into two main components:
- Step 1: Train the Image Style Translation Network (ISTN)
```
cd ISTN
python train.py
```
- Step 2: Generate Translated Images
```
python test.py
```
The output images will be stored in the designated folder. Use them to replace trainA and testA images before training the segmentation model.
- Step 3: Train the Invariance Adaptation Segmentation Network (IASN)
```
cd IASN
python train_iasn.py
```
- Step 4: Evaluate the Model
```
python evaluate.py
```

## 🙏 Acknowledgments
We gratefully acknowledge the following open-source projects and their authors, which our work builds upon:
- [PointCloudUDA](https://github.com/sulaimanvesal/PointCloudUDA)
- [TVT](https://github.com/uta-smile/TVT)