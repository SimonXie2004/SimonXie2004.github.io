---
title: Multi-Resolution Image Blending
tags: 
- Computer Vision
- Image Processing
category: UCB-CV-Project
mathjax: true
date: 2024-09-23 14:33:52
header_image: /images/Multi-Resolution_Blending/head.png
abstract: UC-Berkeley 24FA CV Project 2 - Fun with Sobel, Gaussian Filters and Image Pyramids/Stacks; Blend Images by Band-Pass Filters.
---

> UC-Berkeley 24FA CV Project 2:  
> Fun with Sobel, Gaussian Filters and Image Pyramids/Stacks; Blend Images by Band-Pass Filters.

## Roadmap

This project involves three parts:

1. Image Derivatives
2. Image Filtering & Hybrid: A Reproduction of [Paper](http://cvcl.mit.edu/hybridimage.htm) (SIGGRAPH 2006)
3. Multi-Resolution Blending: A Reproduction of [Paper](https://persci.mit.edu/pub_pdfs/spline83.pdf)

## Part I: Image Derivatives

### Finite Difference Operators

<img src="/images/Multi-Resolution_Blending/image-20240923020522694.png" style="zoom:80%;" />

1. The simplest way to compute derivatives on images is by convolving the image with finite difference operators (FODs). Take $D_x = \begin{bmatrix}1 & -1 \end{bmatrix}$ as an example, for each pixel in the resulting image (excepting those on the edge), they are equal to the original pixel minus its neighbor to the right. This operator is hence sensitive to **vertical** gradients. 

   Similarly, $D_y = \begin{bmatrix}1 \\ -1 \end{bmatrix}$ is sensitive to **horizontal** gradients.

2. Here is an example of convolving the image with FOD operator:

   <img src="/images/Multi-Resolution_Blending/image-20240923020201367.png" />

3. Furthermore, we can **binarize** the image. An appropriate threshold will filter all the **edges** from this gradient image.

4. Here is an example of binarizing the gradient image:

   <img src="/images/Multi-Resolution_Blending/image-20240923020354958.png" />

   > Details: Suppose pixel $\in [0, 1]$. 
   >
   > The threshold is `np.logical_or(im > 0.57, im < 0.43)`

### Derivative of Gaussian Filter

1. From the upper result, we notice that it's rather noisy. This is usually introduced by those **high-frequency components** in the image, which doesn't contribute to edges but literally causes a change in gradient. Hence, we introduce **Gaussian Filters** to get rid of those noises.

2. Here is an example of implementing Gaussian Filters before calculating image gradients:

   <img src="/images/Multi-Resolution_Blending/image-20240923020810400.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923020815937.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923021509660.png" />

   > The threshold is: `np.logical_or(im > 0.52, im < 0.42)`

3. As the **difference**, implementing Gaussian filters can remove those sparkling noises in the picture. (Especially for the lawn, the effect is obvious). However, the **tradeoff** is that some edges are weakened. Take a look at those houses further away, their edges are simply omitted because there isn't a strong gradient at all.

4. Since convolution operation is **commutative**, i.e. 
   $$
   Img * Gaus * D_x = Img * (Gaus * D_x)
   $$
   we can introduce **Derivative of Gaussian Filter**, defined as $Gaus * D_x$ or $Gaus * D_y$. We can verify that convolving the image with Gaus & Dx is equivalent to convolving it with DoG filter.

   <img src="/images/Multi-Resolution_Blending/image-20240923140249941.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923140254947.png" />

## Part II: Image Filtering

> A Reproduction of [Paper](http://cvcl.mit.edu/hybridimage.htm) (SIGGRAPH 2006)

### Image Sharpening

1. By Furrier Transform, we can observe that Gaussian Filter is essentially a **Low-Pass Filter**. Therefore, if we subtract the low-frequency component of an image from its original version, we can get those **High-Frequency Components**, which includes all edges, textures, etc. 

2. We may blend these high frequency components with the original image itself by formula $a * Image + (1-a) * HighFreq$. This will implement the process of **Image Sharpening**

   <img src="/images/Multi-Resolution_Blending/image-20240923140912109.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923140923671.png" />

3. For evaluation, we will also **blur** the image first, and then **sharpen** it again to see what will happen. Here are some results:

   <img src="/images/Multi-Resolution_Blending/image-20240923141044160.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923141049907.png" />

   From the results, we may observe that sharpening may make the image look "sharper", but doesn't actually bring all the tiny textures back. Also, this may introduce **reconstruction errors**, where some lines seems thicker than their original version, which is introduced by gaussian filters.

   This is because that the gaussian filters have already removed high-frequency components from the picture. If we re-sharpen it, we are actually enhancing the high-frequency components of the blurry image, which is not the high-frequency component of the original image. Hence error is introduced.

### Hybrid Images

<img src="/images/Multi-Resolution_Blending/image-20240923141834965.png" />

1. Here is a fun fact about frequency: High frequency of the signal may dominate perception when you are close to a image, but only low frequency of signal can be seen at a distance. 

2. Hence, we may come up with some cool ideas: what about **blending** the high frequency of one signal with the low frequency of another signal, to get a image that look like A seeing from a distance but seems like B seeing closely?

3. From previous sections, we have already known how to extract the high/low frequency components from a image. Hence, lets blend them together!

4. Here are some results:

   <img src="/images/Multi-Resolution_Blending/image-20240923141810780.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923143056093.png" />

5. More Results

   <img src="/images/Multi-Resolution_Blending/image-20240923141922089.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923141834965.png" />

6. Failure Result Example

   <img src="/images/Multi-Resolution_Blending/image-20240923141942239.png" />

   Explanation: It's a "perceptual" failure. In our cognition, the surface texture of the bread is not like a steak's, while the steak's color isn't yellow as well. Hence they don't "blend" together very well. Also, the HP Filter is not good at extracting the tiny textures on the steak. This also contributed to failure.

## Part III: Multi-Resolution Blending

> A Reproduction of [Paper](https://persci.mit.edu/pub_pdfs/spline83.pdf)

Intuitive: if we simply implement alpha-blending on images, the results will look strange with unnatural transitions. How can we come up with a method to blend them together better?

<img src="/images/Multi-Resolution_Blending/image-20240923142418940.png" alt="image-20240923142418940" style="zoom: 80%;" />

### Gaussian and Laplacian Stacks

1. First, we introduce the Gaussian and Laplacian stack. This is to represent the image's different frequency components in a **hierarchal** way. For each level of image, we blur it and push it into stack.

   <img src="/images/Multi-Resolution_Blending/image-20240923142514805.png" style="zoom: 70%;"/>

   > Image cited from CS180 FA24, UC Berkeley, Alexei Efros.

### Multi-Resolution Blending

1. Here are some results (Gaussian Stacks & Laplacian Stacks):

   <img src="/images/Multi-Resolution_Blending/image-20240923142631089.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142637696.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142642875.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142646692.png" />

3. Also, for the convenience of blending, we also implement a Gaussian Stack on a mask.

   <img src="/images/Multi-Resolution_Blending/image-20240923142711330.png" />

4. Everything has been prepared! We may simply blend images together, according to the mask value.

   <img src="/images/Multi-Resolution_Blending/image-20240923142735509.png" />

   And here comes the juicy ora-pple! (Need to **collapse** the Laplacian Stack by adding all layers together)

   <img src="/images/Multi-Resolution_Blending/image-20240923142748599.png" />

5. Here are some of more results:

   **Sun-Moon**:

   <img src="/images/Multi-Resolution_Blending/image-20240923142844537.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142904901.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142909568.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142914312.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142919421.png" />

   **Coca-Pepsi**:

   <img src="/images/Multi-Resolution_Blending/image-20240923142937013.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142941444.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142945690.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142949961.png" />

   <img src="/images/Multi-Resolution_Blending/image-20240923142955127.png" />
