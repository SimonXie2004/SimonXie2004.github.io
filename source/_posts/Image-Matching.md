---
title: Image-Matching
tags: CV
date: 2024-09-09 15:56:02
# mathjax: true
---


# Image Matching

> UC-Berkeley CV Project 1

## 0. Introduction

In project 1, we were provided with a series of photographs related to the Russian Empire. These images were given in separate RGB channels. Unfortunately, they are not perfectly aligned, and due to their age, there are also some color issues. Therefore, the task of project 1 is to restore these photos, which involves aligning them first and then making some color corrections.

> Example Images:
> <img src="/images/Image-Matching/tobolsk.jpg" alt="tobolsk" style="zoom:110%;" />

## 1. Image Aligning

### 1.1 Preprocessing: Removing Borders

This part is relatively simple. We start by calculating the average values both horizontally and vertically, resulting in a `[H, 1]` matrix and a `[1, W]` matrix. Next, we remove the outermost white border and the second outermost black border using a threshold. Specifically, if the average value of a line is greater than 250 (an example threshold), we remove the entire line; otherwise, we keep it.

### 1.2 Single-Layer Image Aligning

Firstly, we align low-resolution images by shifting them pixel by pixel. This is rather brute-force, but the low-resolution makes the calculation time acceptable.

Also, after shifting the pixels, a good metric to asses the similarity of two pictures is crucial. Traditional Euclidian Distances aren't good ideas, since different channels don't necessarily have same features like average brightness, variance, etc. Here we choose SSIM as our assessment function. SSIM is defined as a product of 3 parts: Lightness part, Contrast part and Structural part:

<img src="/images/Image-Matching/SSIM.png" alt="The SSIM Index for Image Quality Assessment" style="zoom: 90%;" />

Here is a visualized result why SSIM is better than Euclidean Distance:

<img src="/images/Image-Matching/maxmin_SSIM.gif" alt="The SSIM Index for Image Quality Assessment" style="zoom: 70%;" />

> Figure 1: equal-MSE hypersphere from [cns.nyu.edu/~lcv/ssim/](https://www.cns.nyu.edu/~lcv/ssim/)

### 1.3 Multiscale Image Aligning

Although the basic method works for low-resolution images, it usually fails for larger images due to the workload increasing with $O(n^2)$. To address this, we use an image pyramid approach, which combines both low-resolution (down-sampled) and high-resolution (original) versions of the image. This allows us to handle the problem iteratively: we first align the smallest image, then transfer the resulting offsets to progressively larger images, aligning each one in turn.

Specifically, this approach uses a stack structure. Initially, a `while` loop `push`es images, each halved in size from the previous one, into the `stack`. In each iteration, the loop `pop`s the last image from the stack and aligns it. The resulting offset is then doubled and applied to the next higher-resolution image in the subsequent loop iteration.

<img src="/images/Image-Matching/800px-Image_pyramid.svg.png" alt="Pyramid (image processing) - Wikipedia" style="zoom:70%;" />

> Figure 2: image pyramid from [Pyramid (image processing) - Wikipedia](https://en.wikipedia.org/wiki/Pyramid_(image_processing))

### 1.4 Interim Result Gallery

<img src="/images/interm-results/cathedral.jpg" width="60%" style="display:inline-block; margin-right: 10px;" /> <img src="/images/interm-results/church.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> <img src="/images/interm-results/emir.jpg" width="30%" style="display:inline-block;" />
<img src="/images/interm-results/harvesters.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> <img src="/images/interm-results/icon.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> <img src="/images/interm-results/lady.jpg" width="30%" style="display:inline-block;" />
<img src="/images/interm-results/melons.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> <img src="/images/interm-results/monastery.jpg" width="60%" style="display:inline-block; margin-right: 10px;" /> <img src="/images/interm-results/onion_church.jpg" width="30%" style="display:inline-block;" />
<img src="/images/interm-results/sculpture.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> <img src="/images/interm-results/self_portrait.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> <img src="/images/interm-results/three_generations.jpg" width="30%" style="display:inline-block;" />
<img src="/images/interm-results/tobolsk.jpg" width="60%" style="display:inline-block; margin-right: 10px;" /> <img src="/images/interm-results/train.jpg" width="30%" style="display:inline-block; margin-right: 10px;" />

> Figure 3: my interim results with multi-layer matching.

## 2. Post-Processing on Images

### 2.1 Auto White-Balancing

This approach is similar to the gray-world assumption but includes a few modifications to produce more reasonable results.

Specifically, a `GRAY_SCALE_THRESHOLD` is defined to filter pixels. Only pixels with channel values close to the mean are used in calculating the gain factors, ensuring that only effective pixels are considered.

Additionally, when calculating the mean values, the outer 10% of the image border is excluded to avoid potential biases.

### 2.2 Auto Contrasting

This is a straightforward algorithm. We select the maximum value among the three channels and remap the entire image accordingly. It’s important to note that we do not apply auto-contrasting separately to each channel, as doing so would disrupt the white balance established in section 2.1.

### 2.3 Ablation Study

![img1](/images/interm-results/church.jpg) ![img2](/images/final-results/church.jpg)

> Figure 4: Auto-contrasting & white-balancing (top: off; bottom: on)

![img1](/images/interm-results/sculpture.jpg) ![img2](/images/final-results/sculpture.jpg)

> Figure 5: Auto-contrasting & white-balancing (top: off; bottom: on)

In comparison, the images on the left are overly blue, green, or red (from top to bottom). The auto-contrasting and white-balancing algorithms have made slight adjustments to improve their appearance.

## 3. Final Results

<img src="/images/final-results/cathedral.jpg" width="70%" style="display:inline-block; margin-right: 10px;" /> 

> Cathedral.jpg: Gshift: (2, 5); Rshift: (3, 12)

<img src="/images/final-results/church.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> 

> Church.jpg: Gshift: (4, 25); Rshift: (-4, 58)

<img src="/images/final-results/emir.jpg" width="30%" style="display:inline-block;" />

> Emir.jpg: Gshift: (23, 49); Rshift: (41, 106)

<img src="/images/final-results/harvesters.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> 

> Harvesters.jpg: Gshift: (17, 59); Rshift: (14, 123)

<img src="/images/final-results/icon.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> 

> Icon.jpg: Gshift: (17, 40); Rshift: (23, 90)

<img src="/images/final-results/lady.jpg" width="30%" style="display:inline-block;" />

> Lady.jpg: Gshift: (9, 55); Rshift: (12, 118)

<img src="/images/final-results/melons.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> 

> Melon.jpg: Gshift: (10, 81); Rshift: (13, 177)

<img src="/images/final-results/monastery.jpg" width="60%" style="display:inline-block; margin-right: 10px;" /> 

> Monastery.jpg: Gshift: (2, -3); Rshift: (2, 3)

<img src="/images/final-results/onion_church.jpg" width="30%" style="display:inline-block;" />

> Onion_church.jpg: Gshift: (27, 51); Rshift: (37, 108)

<img src="/images/final-results/sculpture.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> 

> Sculpture.jpg: Gshift: (-11, 33); Rshift: (-27, 140)

<img src="/images/final-results/self_portrait.jpg" width="30%" style="display:inline-block; margin-right: 10px;" /> 

> Self_portrait.jpg: Gshift: (29, 78); Rshift: (37, 175)

<img src="/images/final-results/three_generations.jpg" width="30%" style="display:inline-block;" />

> Three_generations.jpg: Gshift: (14, 52); Rshift: (11, 111)

<img src="/images/final-results/tobolsk.jpg" width="60%" style="display:inline-block; margin-right: 10px;" /> 

> Tobolsk.jpg: Gshift: (3, 3); Rshift: (3, 7)

<img src="/images/final-results/train.jpg" width="30%" style="display:inline-block; margin-right: 10px;" />

> Train.jpg: Gshift: (5, 41); Rshift: (31, 85)

## 4. Extended Results

<img src="/images/final-results/lugano.jpg" width="80%" style="display:inline-block; margin-right: 10px;" /> 

<img src="/images/final-results/lugano2.jpg" width="80%" style="display:inline-block; margin-right: 10px;" /> 

> Lugano.jpg: Gshift: (-2, 4); Rshift: (-3, 9)

> Lugano2.jpg: Gshift: (1, 3); Rshift: (4, 8)
