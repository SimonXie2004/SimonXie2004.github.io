---
title: Image Matching & Colorizing
tags: 
- Computer Vision
- SSIM
- Image Auto Balancing
category: UCB-CV-Project
date: 2024-09-09 15:56:02
mathjax: true
header_image: /images/Image-Matching/head.png
abstract: UC-Berkeley 24FA CV Project 1 - Colorizing the Images of the Russian Empire by Image Matching
---

> UC-Berkeley 24FA CV Project 1: Colorizing the Images of the [Russian Empire](https://www.loc.gov/collections/prokudin-gorskii/?st=grid)

## RoadMap

This project involves two parts:  
1. **Colorize** the image by aligning the **mismatched** RGB channels separately.  
2. Fix the color issues with **Auto-Contrasting & WhiteBalancing**  

<img src="/images/Image-Matching/rgb.png" style="zoom: 110%"/>

## Background

Sergei Mikhailovich Prokudin-Gorskii (1863-1944) was a man well ahead of his time. Convinced, as early as 1907, that color photography was the wave of the future, he won Tzar's special permission to travel across the vast Russian Empire and take color photographs of everything he saw including the only color portrait of Leo Tolstoy. And he really photographed everything: people, buildings, landscapes, railroads, bridges... thousands of color pictures!  
His idea was simple: record three exposures of every scene onto a glass plate using a red, a green, and a blue filter. Never mind that there was no way to print color photographs until much later -- he envisioned special projectors to be installed in "multimedia" classrooms all across Russia where the children would be able to learn about their vast country. Alas, his plans never materialized: he left Russia in 1918, right after the revolution, never to return again.  
Luckily, his RGB glass plate negatives, capturing the last years of the Russian Empire, survived and were purchased in 1948 by the Library of Congress. The LoC has recently digitized the negatives and made them available on-line.  

<img src="/images/Image-Matching/teaser.png" style="zoom: 60%"/>

## Image Aligning

### Preprocessing: Removing Borders

1. We start by calculating the average values both horizontally and vertically, resulting in a `[H, 1]` matrix and a `[1, W]` matrix. 
2. Next, we remove the outermost white border and the second outermost black border using a threshold. 

### Single-Layer Image Aligning

1. Firstly, we align low-resolution images by exhaustively searching over a window of possible displacements, score each one using some image matching metric, and take the displacement with the best score. This is rather brute-force, but the low-resolution makes the calculation time acceptable.

2. A good metric to asses the similarity of two pictures is crucial. Traditional Euclidian Distances aren't good ideas, since different channels don't necessarily have same features like average brightness, variance, etc. 
   
3. Here we choose SSIM as our assessment function. SSIM is defined as a product of 3 parts: Lightness part, Contrast part and Structural part. Here is a visualized result why SSIM is better than Euclidean Distance:

<img src="/images/Image-Matching/maxmin_SSIM.jpg" alt="The SSIM Index for Image Quality Assessment" style="zoom: 55%;" />

> Figure 1: equal-MSE hypersphere from [cns.nyu.edu/~lcv/ssim/](https://www.cns.nyu.edu/~lcv/ssim/)

$$
\text{SSIM}(x, y) = l(x, y)^\alpha \cdot c(x, y)^\beta \cdot s(x, y)^\gamma \\
$$

$$
\begin{aligned}
    l(x, y) &= \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1} \\
    c(x, y) &= \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2} \\
    s(x, y) &= \frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}
\end{aligned}
$$

> Remark: Introduce the constant $C_1, C_2, C_3$ to avoid zero division. Usually a tiny number.


### Multiscale Image Aligning

Although the basic method works for low-resolution images, it usually fails for larger images due to the workload increasing with $O(n^2)$. To address this, we use an image pyramid approach, which combines both low-resolution (down-sampled) and high-resolution (original) versions of the image. This allows us to handle the problem iteratively: we first align the smallest image, then transfer the resulting offsets to progressively larger images, aligning each one in turn. Specifically, this approach uses a stack structure.

1.  Initially, a `while` loop `push`es images, each halved in size from the previous one, into the `stack`. 
2.  In each iteration, the loop `pop`s the last image from the stack and aligns it. 
3.  The resulting offset is then doubled and applied to the next higher-resolution image in the subsequent loop iteration.

<img src="/images/Image-Matching/pyramid.png" alt="Pyramid (image processing) - Wikipedia" style="zoom:70%;" />

> Figure 2: image pyramid from [Pyramid (image processing) - Wikipedia](https://en.wikipedia.org/wiki/Pyramid_(image_processing))

### Interim Result Gallery

<table style="max-width: 100%; table-layout: fixed; width: 100%;">
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/emir.jpg" alt="emir.jpg" width="200"/>
        <figcaption>Gshift: (23, 49) <br> Rshift: (41, 106)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/cathedral.jpg" alt="cathedral.jpg" width="200"/>
        <figcaption>Gshift: (2, 5) <br> Rshift: (3, 12)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/church.jpg" alt="church.jpg" width="200"/>
        <figcaption>Gshift: (4, 25) <br> Rshift: (-4, 58)</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/harvesters.jpg" alt="harvesters.jpg" width="200"/>
        <figcaption>Gshift: (17, 59) <br> Rshift: (14, 123)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/icon.jpg" alt="icon.jpg" width="200"/>
        <figcaption>Gshift: (17, 40) <br> Rshift: (23, 90)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/lady.jpg" alt="lady.jpg" width="200"/>
        <figcaption>Gshift: (9, 55) <br> Rshift: (12, 118)</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/melons.jpg" alt="melons.jpg" width="200"/>
        <figcaption>Gshift: (10, 81) <br> Rshift: (13, 177)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/monastery.jpg" alt="monastery.jpg" width="200"/>
        <figcaption>Gshift: (2, -3) <br> Rshift: (2, 3)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/onion_church.jpg" alt="onion_church.jpg" width="200"/>
        <figcaption>Gshift: (27, 51) <br> Rshift: (37, 108)</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/sculpture.jpg" alt="sculpture.jpg" width="200"/>
        <figcaption>Gshift: (-11, 33) <br> Rshift: (-27, 140)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/self_portrait.jpg" alt="self_portrait.jpg" width="200"/>
        <figcaption>Gshift: (29, 78) <br> Rshift: (37, 175)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/three_generations.jpg" alt="three_generations.jpg" width="200"/>
        <figcaption>Gshift: (14, 52) <br> Rshift: (11, 111)</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/tobolsk.jpg" alt="tobolsk.jpg" width="200"/>
        <figcaption>Gshift: (3, 3) <br> Rshift: (3, 7)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/train.jpg" alt="train.jpg" width="200"/>
        <figcaption>Gshift: (5, 41) <br> Rshift: (31, 85)</figcaption>
      </figure>
    </td>
  </tr>
</table>

> Figure 3: My interim results with multi-layer matching.  
> The G & R Value below the image are shift values of green & red channel.

## Post-Processing on Images

### Auto White-Balancing

In this project, we implement the gray world algorithm, which is a classic auto white balance algorithm that estimates the illuminant of an image by assuming that the average color of the world is gray. The algorithm is based on the idea that the average reflectance of surfaces in the world is achromatic, or gray.

> Remark: when calculating the mean values, the outer 10% of the image border is excluded to avoid potential biases (since the image itself is damaged).

### Auto Contrasting

This is a straightforward algorithm. We select the minimum & maximum value $\min, \max$ among the three channels and remap the entire image to $[0, 255]$ accordingly. It’s important to note that we do not apply auto-contrasting separately to each channel, as doing so would disrupt the white balance established in section 2.1.

### Ablation Study

<table style="max-width: 100%; table-layout: fixed; width: 100%;">
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/sculpture.jpg" alt="sculpture.jpg" width="300"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/sculpture.jpg" alt="sculpture_balanced.jpg" width="300"/>
      </figure>
    </td>
  </tr>
</table>

> Figure 4: Auto-contrasting & white-balancing (left: off; right: on)

<table style="max-width: 100%; table-layout: fixed; width: 100%;">
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/interm-results/church.jpg" alt="church.jpg" width="300"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/church.jpg" alt="church_balanced.jpg" width="300"/>
      </figure>
    </td>
  </tr>
</table>

> Figure 5: Auto-contrasting & white-balancing (top: off; bottom: on)

In comparison, the images on the left are overly blue, green, or red. The auto-contrasting and white-balancing algorithms have made slight adjustments to improve their appearance.

## Final Results

> Here are all **mandatory** final results.

<table style="max-width: 100%; table-layout: fixed; width: 100%;">
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/emir.jpg" alt="emir.jpg" width="200"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/cathedral.jpg" alt="cathedral.jpg" width="200"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/church.jpg" alt="church.jpg" width="200"/>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/harvesters.jpg" alt="harvesters.jpg" width="200"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/icon.jpg" alt="icon.jpg" width="200"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/lady.jpg" alt="lady.jpg" width="200"/>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/melons.jpg" alt="melons.jpg" width="200"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/monastery.jpg" alt="monastery.jpg" width="200"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/onion_church.jpg" alt="onion_church.jpg" width="200"/>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/sculpture.jpg" alt="sculpture.jpg" width="200"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/self_portrait.jpg" alt="self_portrait.jpg" width="200"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/three_generations.jpg" alt="three_generations.jpg" width="200"/>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/tobolsk.jpg" alt="tobolsk.jpg" width="200"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/train.jpg" alt="train.jpg" width="200"/>
      </figure>
    </td>
  </tr>
</table>

## More Results

> Here are some results that I appreciate!

<table style="max-width: 100%; table-layout: fixed; width: 100%;">
  <tr>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/lugano.jpg" alt="lugano.jpg" width="300"/>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/images/Image-Matching/final-results/lugano2.jpg" alt="lugano2.jpg" width="300"/>
      </figure>
    </td>
  </tr>
</table>
