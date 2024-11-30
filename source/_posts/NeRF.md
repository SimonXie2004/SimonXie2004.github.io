---
title: Implementing NeRF
mathjax: true
date: 2024-11-29 17:09:39
tags:
- Computer Vision
- Deep Learning
- New View Synthesis
category: UCB-CV-Project
header_image: /images/NeRF/teaser.png
abstract: UC Berkeley CV Project 6 - Implementing Neural Radiance Fields
---

> UC Berkeley CV Project 6 - Implementing Neural Radiance Fields

<img src="/images/NeRF/final.gif"/>

## Part I: Fit a Neural Field to a 2D Image

### Introduction

We know that we can use a Neural Radiance Field (NeRF) ($F:\{x,y,z,\theta, \phi\}\rightarrow\{r,g,b,\sigma\}$) to represent a 3D space. But before jumping into 3D, let's first get familiar with NeRF using a 2D example. In fact, since there is no concept of radiance in 2D, the Neural Radiance Field falls back to just a Neural Field ($F:\{u,v\}\rightarrow \{r,g,b\}$) in 2D, in which $\{u,v\}$ is the pixel coordinate. Hence, in this section, we will create a neural field that can represent a 2D image and optimize that neural field to fit this image.

+ MLP: We start from the following MLP structure, but the number of hidden layers (Linear 256 in the graph) is configurable as parameter `num_hidden_layers`. We will show the effect of different `num_hidden_layers` later.

  <img src="/images/NeRF/mlp_img.jpg"/>

+ Sinusoidal Positional Encoding (PE): PE is an operation that you apply a serious of sinusoidal functions to the input cooridnates, to expand its dimensionality (See equation 4 from [this paper](https://arxiv.org/pdf/2003.08934.pdf) for reference). Note we also additionally keep the original input in PE, so the complete formulation is
  $$
  PE(x) = \left[ x, \sin\left( 2^0 \pi x \right), \cos\left( 2^0 \pi x \right), \cdots, \sin\left( 2^{L-1} \pi x \right), \cos\left( 2^{L-1} \pi x \right) \right]
  $$

### Experiments

For the following results, we let:

+ Learning Rate = `1e-3`
+ Number of Hidden Neurons = `256`
+ Batch Size = "full image" (meaning that each batch is consisted of all pixels from the original image)

And, the following params are subject to different experimental values:

+ N = num_hidden_layers
+ L = maximum positional encoding power

<img src="/images/NeRF/big_bad_wolf.jpg" alt="gt" style="zoom:65%;" />

<table>
    <tr>
        <td rowspan="2" width="10%"><strong>N=3<br>L=10</strong></td>
        <td><img src="/images/NeRF/2dnerf_config1_epoch64.png" alt="图片1" width="100%"></td>
        <td><img src="/images/NeRF/2dnerf_config1_epoch128.png" alt="图片2" width="100%"></td>
        <td><img src="/images/NeRF/2dnerf_config1_epoch512.png" alt="图片3" width="100%"></td>
        <td><img src="/images/NeRF/2dnerf_config1_epoch4096.png" alt="图片4" width="100%"></td>
    </tr>
    <tr>
        <td>Epoch=64, PSNR=16.72</td>
        <td>Epoch=128, PSNR=21.30</td>
        <td>Epoch=512, PSNR=25.58</td>
        <td>Epoch=4096, PSNR=28.43</td>
    </tr>
    <tr>
        <td rowspan="2"><strong>N=5<br>L=20</strong></td>
        <td><img src="/images/NeRF/2dnerf_config2_epoch64.png" alt="图片1" width="100%"></td>
        <td><img src="/images/NeRF/2dnerf_config2_epoch128.png" alt="图片2" width="100%"></td>
        <td><img src="/images/NeRF/2dnerf_config2_epoch512.png" alt="图片3" width="100%"></td>
        <td><img src="/images/NeRF/2dnerf_config2_epoch4096.png" alt="图片4" width="100%"></td>
    </tr>
    <tr>
        <td>Epoch=64, PSNR=15.97</td>
        <td>Epoch=128, PSNR=19.95</td>
        <td>Epoch=512, PSNR=25.39</td>
        <td>Epoch=4096, PSNR=29.76</td>
    </tr>
</table>

<table style="width: 100%; text-align: center; border-collapse: collapse;">
    <tr>
        <td rowspan="2" width="10%">PSNR</td>
        <td><img src="/images/NeRF/2dnerf_config1_psnr.png" alt="图片1" width="100%"></td>
        <td><img src="/images/NeRF/2dnerf_config2_psnr.png" alt="图片1" width="100%"></td>
    </tr>
    <tr>
        <td>PSNR Curve for config 1 <strong>(N=3, L=10)</strong></td>
        <td>PSNR Curve for config 2 <strong>(N=5, L=20)</strong></td>
    </tr>
</table>

The results align well with our expectations: Larger networks and deeper positional encodings make convergence more challenging, but ultimately lead to better performance.

Furthermore, we aim to demonstrate the effectiveness of our positional encoding. To achieve this, we conduct an ablation experiment to visualize the differences, where both images have same neural network but different positional encoding powers.

<img src="/images/NeRF/texture-17329288779126.jpg" alt="texture" style="zoom:65%;" />

<table>
    <tr>
        <td rowspan="2" width="70px">N=3</td>
        <td><img src="/images/NeRF/2dnerf_config3_epoch4096_zoomin.png" alt="图片1" width="100%"></td>
        <td><img src="/images/NeRF/2dnerf_config4_epoch4096_zoomin.png" alt="图片2" width="100%"></td>
    </tr>
    <tr>
        <td><strong>L=5</strong>, PSNR=22.46</td>
        <td><strong>L=20</strong>, PSNR=28.76</td>
    </tr>
</table>

<table style="width: 100%; text-align: center; border-collapse: collapse;">
    <tr>
        <td rowspan="2" width="70px">PSNR</td>
        <td><img src="/images/NeRF/2dnerf_config3_psnr.png" alt="图片1" width="100%"></td>
        <td><img src="/images/NeRF/2dnerf_config4_psnr.png" alt="图片1" width="100%"></td>
    </tr>
    <tr>
        <td>PSNR Curve for config 1 <strong>(L=5)</strong></td>
        <td>PSNR Curve for config 2 <strong>(L=20)</strong></td>
    </tr>
</table>

## Part II: 3D Neural Radiance Field!

### Introduction

**Neural Radiance Fields (NeRF)** is a deep learning framework introduced for generating photorealistic 3D scenes from a sparse set of 2D images. NeRF models the scene as a continuous volumetric scene representation, where a neural network is trained to predict the color and density of points in 3D space. It has shown success in synthesizing realistic novel views of a scene, particularly for photo-realistic rendering, by leveraging the power of neural networks.

### Dataloader

In NeRF, the core idea is that we want to compute how light travels through a 3D scene by casting rays from a camera into the scene. Each ray represents a potential view of a point in the scene, and it passes through the 3D volume. (Hence, this requires that we know each camera's position and looking-direction, i.e. the cameras are **calibrated**)

- **Camera rays**: During training, rays are cast from the camera's viewpoint through each pixel of the image, into the scene. Each pixel in the image corresponds to one ray in 3D space. The parameter $r_o$ and $r_d$ can be calculated as follows:
  $$
  r_o = \mathbf{t} \\
  r_d = \frac{\mathbf{X_w} - \mathbf{r_o}}{\| \mathbf{X_w} - \mathbf{r_o} \|}
  $$

  (Suppose that the calibrated extrinsics are `c2w` matrices)

Here’s a simple illustration of this step:

<img src="/images/NeRF/ray_vis1.png"  >

<img src="/images/NeRF/ray_vis2.png"  >

### Scene Representation (MLP)

NeRF represents a 3D scene as a continuous function that predicts the color and density at any given 3D point, parameterized by spatial coordinates and viewing directions. Specifically, a neural network $f_{\theta}$ learns to predict the radiance (color) and volume density at a point $\mathbf{x} = (x, y, z)$ and view direction $\mathbf{d}$:
$$
f_\theta(\mathbf{x},\mathbf{d})=(\hat{C},\sigma)
$$
where $\hat{C}$ is the color and $\sigma$ is the volume density.

Here, we implement the following network architecture as our NeRF scene representation. Remark that by concatenation of x in the middle of linear layers, we can let the network "remember" the positional information.

![](/images/NeRF/mlp_nerf.png)

### Volumn Rendering

The core volume rendering equation is as follows:
$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d})dt \text{, where } T(t) = \exp(-\int_{t_n}^t\sigma(\mathbf{r}(s)))ds
$$
This fundamentally means that at every small step $dt$ along the ray, we add the contribution of that small interval $[t,t+dt]$ to that final color, and we do the infinitely many additions if these infintesimally small intervals with an integral.

The discrete approximation (thus tractable to compute) of this equation can be stated as the following:
$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^N(1 - \exp(-\sigma_i\delta_i))\mathbf{c}_i \text{, where } T_i = \exp(-\sum_{j=1}^{i-1}\sigma_j\delta_j)
$$

### Experiments

<table>
    <tr>
        <td><img src="/images/NeRF/nerf_lego_epoch1024.png" alt="图片1" width="100%"></td>
        <td><img src="/images/NeRF/nerf_lego_epoch2048.png" alt="图片2" width="100%"></td>
        <td><img src="/images/NeRF/nerf_lego_epoch4096.png" alt="图片3" width="100%"></td>
        <td><img src="/images/NeRF/nerf_lego_epoch8192.png" alt="图片4" width="100%"></td>
        <td><img src="/images/NeRF/nerf_lego_epoch40960.png" alt="图片1" width="100%"></td>
    </tr>
    <tr>
        <td>Epoch 1024</td>
        <td>Epoch 2048</td>
        <td>Epoch 4096</td>
        <td>Epoch 8192</td>
        <td>Epoch 40960</td>
    </tr>
</table>

![](/images/NeRF/nerf_lego_psnr.png)

And the final result on test set is:

![](/images/NeRF/final.gif)

