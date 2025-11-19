# RetinaLiteNet Implementation

RetinaLiteNet is a lightweight transformer-based CNN designed for retinal feature segmentation. This repository contains a reimplementation of the methodology described in the original paper, focusing on accurate and efficient segmentation of retinal blood vessels and the optic disc.

The model combines convolutional layers with multi-head self-attention in the encoder to capture both local and global features. A Convolutional Block Attention Module (CBAM) in the decoder helps refine segmentation outputs. This architecture achieves high accuracy while remaining computationally efficient.

**Original code reference:** [Mehwish4593/RetinaLiteNet](https://github.com/Mehwish4593/RetinaLiteNet)

## Features

- Efficient multitask segmentation of retinal vessels and optic disc  
- Lightweight model: small memory footprint and low FLOPs  
- Uses attention mechanisms for enhanced feature representation  

## Datasets

- DRIVE  
- IOSTAR
