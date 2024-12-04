---
title: CS180-Computer Vision Finalterm Review & Cheatsheet
mathjax: true
date: 2024-12-04 00:40:07
tags:
- Computer Vision
- Cheatsheet
category:
- UCB-CV
header_image:
abstract: UC-Berkeley 24FA CV Final Term Review & Cheatsheet
---

> Finalterm Review & Cheatsheet for  
> [CS 180 Fall 2024 | Intro to Computer Vision and Computational Photography](https://www2.eecs.berkeley.edu/Courses/CS180/)
>
> Author: [SimonXie2004.github.io](https://simonxie2004.github.io)

<img src="/images/CV-Cheatsheet-Finalterm/logo.png" alt="CS180 Logo" style="zoom:100%;" />

## Resources

[Download Cheatsheet (pdf)](/files/CV-Cheatsheet-Finalterm/Finalterm.pdf)

[Download Cheatsheet (pptx)](/files/CV-Cheatsheet-Finalterm/Finalterm.pptx)

[Download Cheatsheet (markdown+images)](/files/CV-Cheatsheet-Finalterm/Finalterm.zip)

## Lec2 Capturing Light

1. Psychophysics of Color: Mean => Hue, Variance => Saturation, Area => Brightness

2. Image Processing Sequence

   1. Auto Exposure -> White Balance -> Contrast -> Gamma

3. White Balancing Algorithms

   1. Grey World: force average color of scene to grey
   2. White World: force brightest object to white

4. Quad Bayer Filters

   1. Why more green pixels? Because human are most sensitive to green light.

5. Color Spaces: RGB, CMY(K), HSV, L\*a\*b\* (Perceptually uniform color space)

6. Image similarity: 

   1. SSD, i.e. L2 Distance
   2. NCC, invariant to local avg and contrast

   $$
   \text{NCC}(I, T) = \frac{\sum_{x,y} (I'(x,y) \cdot T'(x,y))}{\sqrt{\sum_{x,y} I'(x,y)^2 \sum_{x,y} T'(x,y)^2}}
   $$


## Lec 3-4: Pixels and Images

1. Lambertian Reflectance Model:

   1. $(1-\rho)$ absorbed, $\rho$ reflected (either diffusely or specularly) 
   2. Diffuse Reflectance: $I(x) = \rho(x) \cdot \mathbf{S} \cdot \mathbf{N}(x)$

2. Image Acquisition Pipeline:

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241130203542989.png" alt="image-20241130203542989" style="zoom: 60%;" />

3. Image Processing: Gamma Correction

   1. Power-law transformations: $s = c\cdot r^\gamma$
   2. Contrast Stretching: S curve (from input gray level to output)

4. Histograms Matching and Color Transfer

5. Image Filter

   1. Cross Correlation $C = K \times I$ or $C(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y) \cdot K(x+u, y+v)$
   2. Convolution: $C = K * I$ or $C(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y) \cdot K(x-u, y-v)$

6. Example: Gaussian Filter

   1. Rule of Thumb: Half-Width = $3\sigma$

7. Image Sub-sampling: Must **first filter** the image, then subsample (Anti-Aliasing)

8. Image Derivatives: To avoid the effects of **noise**, first **smooth**, then **derivative** (i.e. look for peaks in $\frac{d}{dx}(f*g)$)

   1. This leads to LoG or DoG filters


## Lec 5: Furrier Transform

1. Math:
   1. Furrier Transform: $F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i \omega t} \, dt$
   2. Inverse Transform: $f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i \omega t} \, d\omega$
2. Low pass, High pass, Band pass filters
   1. Details = High-freq components;
   2. Sharpening Details: $f + \alpha (f - f * g) = (1+\alpha)f-af*g = f*((1+\alpha)e-\alpha g)$
   3. Remark that $(1+\alpha)e-\alpha g$ is approximately Laplacian of Gaussian.

## Lec 6: Pyramids Blending

1. Gaussian Pyramids and Laplacian Pyramids (Remember to add lowest freq!)

2. Laplacian Pyramids and Image Blending:

   ![](/images/CV-Cheatsheet-Finalterm/image-20241130212109854.png)

3. Other image algorithms:
   1. Denoising: Median Filter
   2. Lossless Compression (PNG): Huffman Coding
   3. Lossy Compression (JPEG): Block-based Discrete Cosine Transform (DCT)
      1. Compute DCT Coefficients; Coarsely Quantize; Encode (e.g. with Huffman Coding)

## Lec 7-9 Affine Transformations

1. Transform Matrices

   1. Scaling, Shearing and Translation: $S = \begin{bmatrix} a & sh_x & t_x \\ sh_y & b & t_y \\ 0 & 0 & 1\end{bmatrix}$
   2. Rotation: $R = \begin{bmatrix} \cos(\theta) & -\sin(\theta) & 0\\ \sin(\theta) & \cos(\theta) & 0 \\ 0 & 0 & 1 \end{bmatrix}$

2. Calculating Affine Matrix: 
   $$
   \begin{pmatrix} a & b & tx \\ c & d & ty \\ 0 & 0
   & 1 \end{pmatrix} \cdot \begin{pmatrix} x_1 & x_2 & x_3 \\
   y_1 & y_2 & y_3 \\ 1 & 1 & 1 \end{pmatrix} =
   \begin{pmatrix} x_1' & x_2' & x_3' \\ y_1' &
   y_2' & y_3' \\ 1 & 1 & 1 \end{pmatrix}
   $$

3. Bilinear Interpolation:

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241201021913803.png" alt="image-20241201021913803" style="zoom: 60%;" />

4. Delaunay Triangulation: Dual graph of Voroni Diagram

   ![](/images/CV-Cheatsheet-Finalterm/image-20241201022044667.png)

5. Morphing and Extrapolation

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241201022304425.png" alt="image-20241201022304425" style="zoom:60%;" />

## Lec 10: Cameras

1. Pinhole camera model

   1. Defines **Center of Projection (CoP)** and **Image Plane**
   2. Defines **Effective Focal Length** as **d**

2. Camera coordinate systems

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241201135020391.png" alt="image-20241201135020391" style="zoom:100%;" />

3. Projection:

   1. Perspective Projection: $(x, y, z) \rightarrow (-d\frac{x}{z}, -d\frac{y}{z}, -d) \rightarrow (-d\frac{x}{z}, -d\frac{y}{z})$

   2. Orthographic Projection: $(x, y, z) \rightarrow (x, y)$; special case when distance from COP to PP is infinite

   3. Weak Perspective/Orthographic: if $\Delta z << -\bar{z}, (x, y, z) \rightarrow (-mx, -my)$ where $m=-\frac{f}{\bar{z}}$

      Special case when scene depth is small relative to avg. distance from camera

      Equivalent to scale first then orthographic project

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241201135727306.png" alt="image-20241201135727306" style="zoom:60%;" />

   4. Spherical Projection: $(\theta,  \phi) \rightarrow (\theta, \phi, d)$

4. Camera Parameters

   1. Aperture: Bigger aperture = Shallower scene depth, Narrower gate width

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241201140240234.png" alt="image-20241201140240234" style="zoom:60%;" />

   2. Thin Lenses: $\frac{1}{d_o} + \frac{1}{d_i} = \frac{1}{f}$

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241201140150863.png" alt="image-20241201140150863" style="zoom:60%;" />

   3. FOV (Field of View): $\phi = \tan^{-1}(\frac{d}{2f})$

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241201140345133.png" alt="image-20241201140345133" style="zoom:50%;" />

   4. Exposure & Shutter Speed

      1. Example: F5.6+1/30Sec = F11+1/8Sec

   5. Lens Flaws

      1. Chromatic Aberration: Due to wavelength-dependent refractive index, modifies ray-bending and focal length

         <img src="/images/CV-Cheatsheet-Finalterm/image-20241201191455881.png" alt="image-20241201191455881" style="zoom:70%;" />

      2. Radial Distortion

         <img src="/images/CV-Cheatsheet-Finalterm/image-20241201191624915.png" alt="image-20241201191624915" style="zoom:100%;" />

## Lec 11: Perspective Transforms

1. Formula: 
   $$
   H =
   \begin{pmatrix}
   a & b & c \\
   d & e & f \\
   g & h & 1
   \end{pmatrix}
   $$

   $$
   \begin{pmatrix}
   x_1 & y_1 & 1 & 0 & 0 & 0 & -x_1 x_1' &
   -y_1 x_1' \\
   0 & 0 & 0 & x_1 & y_1 & 1 & -x_1 y_1' &
   -y_1 y_1' \\
   x_2 & y_2 & 1 & 0 & 0 & 0 & -x_2 x_2' &
   -y_2 x_2' \\
   0 & 0 & 0 & x_2 & y_2 & 1 & -x_2 y_2' &
   -y_2 y_2' \\
   \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
   & \vdots & \vdots \\
   x_N & y_N & 1 & 0 & 0 & 0 & -x_N x_N' &
   -y_N x_N' \\
   0 & 0 & 0 & x_N & y_N & 1 & -x_N y_N' &
   -y_N y_N'
   \end{pmatrix}
   \cdot
   \begin{pmatrix}
   a \\
   b \\
   c \\
   d \\
   e \\
   f \\
   g \\
   h
   \end{pmatrix}
   =
   \begin{pmatrix}
   x_1' \\
   y_1' \\
   x_2' \\
   y_2' \\
   \vdots \\
   x_N' \\
   y_N'
   \end{pmatrix}
   $$

   Solution: Least Squares, $x = (A^TA)^{-1}A^Tb$

## Lec 12-14: Feature Extraction

1. Feature Detector:

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241201192634463.png" alt="image-20241201192634463" style="zoom: 60%;" />

   Change in appearance of window W for the shift $[u, v]$ is: 
   $$
   E(u, v) = \sum_{(x, y) \in W}[I(x+u, y+v) - I(x, y)]^2
   $$
   Then, we use a First-order Taylor approximation for small motions $[u,v]$:
   $$
   \begin{aligned}
   I(x+u, y+v) &= I(x, y) + I_x u + I_y v + \text{higher order terms}
   \\
   &\approx I(x, y) + I_x u + I_y v \\
   &= I(x, y) + \begin{bmatrix} I_x & I_y \end{bmatrix}
   \begin{bmatrix} u \\ v \end{bmatrix}
   \end{aligned}
   $$

   $$
   \begin{aligned}
   E(u, v) &= \sum_{(x, y) \in W} \left[I(x+u, y+v) - I(x, y)\right]^2
   \\
   &\approx \sum_{(x, y) \in W} \left[I(x, y) + \begin{bmatrix} I_x
   & I_y \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} - I(x,
   y)\right]^2 \\
   &= \sum_{(x, y) \in W} \left(\begin{bmatrix} I_x & I_y
   \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix}\right)^2 \\
   &= \sum_{(x, y) \in W} \begin{bmatrix} u & v \end{bmatrix}
   \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
   \begin{bmatrix} u \\ v \end{bmatrix}
   \end{aligned}
   $$

   This gives us the second moment matrix M, which is a approximate of local change on images.
   $$
   M = \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2
   \end{bmatrix}
   $$
   Here, we calculate this function value as "corner strength":
   $$
   R = \det(M) - k * \text{tr}(M)^2 \text{ or } \det(M)/\text{tr}(M)
   $$
   Remark: for flat areas, both $\lambda_1, \lambda_2$ are small; for edges, one of the $\lambda$ is big; for corners, both are big.

2. Scale Invariant Detection: choose the scale of best corner independently!

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241201193403452.png" alt="image-20241201193403452" style="zoom:60%;" />

3. Feature Selection: ANMS

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241201193438439.png" alt="image-20241201193438439" style="zoom:70%;" />

4. Feature Descriptor (Multi-scale Oriented Patches): 8x8 oriented patch, descripted by (x, y, scale, orientation)

   1. Maybe normalized by $I' = (I-\mu)/\sigma$

5. Matching Feature:

   1. Step 1: Lowe's Trick, match(1-NN) - match(2-NN)
   2. Step 2: RANSAC (random choose 4 points; calc homography; calc outliers; finally select best homography)

6. Further Techniques: Order images to reduce inconsistencies

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241201194226796.png" alt="image-20241201194226796" style="zoom:50%;" />

   Do the loop: match images - order images - match images - ...

7. Optical Flow Algorithm
   $$
   0 = I(x+u, y+v)-H(x, y) \approx [I(x, y) - H(x, y)] + I_xu + I_yv = I_t + \nabla I \cdot [u, v]
   $$
   The component of the flow in the gradient direction is determined.

   The component of the flow parallel to an edge is unknown.

   To have more constraint, consider a bigger window size!
   $$
    \begin{bmatrix} I_x(\mathbf{p}_1) & I_y(\mathbf{p}_1) \\ I_x(\mathbf{p}_2) & I_y(\mathbf{p}_2) \\ \vdots & \vdots \\ I_x(\mathbf{p}_{25}) & I_y(\mathbf{p}_{25}) \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = - \begin{bmatrix} I_t(\mathbf{p}_1) \\ I_t(\mathbf{p}_2) \\ \vdots \\ I_t(\mathbf{p}_{25}) \end{bmatrix} 
   $$
   Solve by least square: (Lukas & Kanade, 1981)
   $$
   (A^T A) \mathbf{d} = A^T \mathbf{b}
   $$

   $$
   \begin{bmatrix}
   \sum I_x I_x & \sum I_x I_y \\
   \sum I_x I_y & \sum I_y I_y
   \end{bmatrix}
   \begin{bmatrix}
   u \\
   v
   \end{bmatrix}
   =
   -
   \begin{bmatrix}
   \sum I_x I_t \\
   \sum I_y I_t
   \end{bmatrix}
   $$

   This is solvable when: no aperture problem.

   How to make it even better? Do it multi-hierachical!

## Lec 15-16: Texture Models

1. Human vision patterns

   1. Pre-attentive vision: parallel, instantaneous (~100--200ms), without scrutiny, independent of the number of patterns, covering a large visual field.
   2. Attentive vision: serial search by focal attention in 50ms steps limited to small aperture.

2. Order statistics of Textures

   1. Textures cannot be spontaneously discriminated if they have the same first-order and second-order statistics of texture features (textons) and differ only in their third-order or higher-order statistics.
   2. First order: mean, var, std, ...
   3. Second order: co-occurence matrix, contrast, ...

3. Introduction: Cells in Retina

   1. Receptive field of a retinal ganglion cell can be modeled as a LoG filter. (Corner Detectors)
   2. Cortical Receptive Fields -> (Line/Edge Detectors)
   3. They are connected just like a CNN network.

4. From Cells to Image Filters: [Filter Banks]

   Detects Statistical unit of texture (texton) in real images:

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241202001928276.png" alt="image-20241202001928276" style="zoom:60%;" />

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241202001941598.png" alt="image-20241202001941598" style="zoom:60%;" />

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241202002150145.png" alt="image-20241202002150145" style="zoom:60%;" />

   Texton summary: from object to bag of "words"; Preliminaries of CNN

5. Image Feature Repr:

   1. Code Words -> Hist Matching

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202002911559.png" alt="image-20241202002911559" style="zoom: 70%;" />

6. Image-2-Image Translation

   1. Target: Depths, Normals, Pixelwise-Segmentation/Labelling, Grey-2-Color, Edges-2-Photo, ...
   2. Wrong exmples:
      1. Stacking $2n+1$ convolutions (receptive fields) for $n$ pixels: too many convolutions!
      2. Extract NxN patches and independently do CNN: requires too much patches
   3. Answer: Encoder+Decoder, Convolutions and Pooling
   4. How about missing details when up-sampling? Copy a high-resolution Version! (U-Net)
   5. How about loss function? L2 don't work for task: image colorization
      1. Use per-pixel multinomial classification! (Some what like a bayes net P(label1|pix1), P(label2|pix1), ...)
      2. $L(\mathbf{\hat{Z}}, \mathbf{Z}) = -\frac{1}{HW}\sum_{h,w}\sum_{q}\mathbf{Z}_{h,w,q}\log(\mathbf{\hat{Z}}_{h,w,q})$ where q is probability of label q; This is a cross-entropy.

7. Universal loss for Img2Img Tasks: GAN

   1. Structure: Input -> Generator -> Generated Im; G(x) -> Discriminator -> Loss (Represented as probability; Suppose D(x) is prob that x is fake)

   2. D's task: $\arg \max_D \mathbb{E}_{x, y}[\log D(G(x)) + \log(1-D(y))]$

   3. G's task:

      1. Tries to synthesize fake images that fool D:  $\arg \min_G \mathbb{E}_{x, y}[\log D(G(x)) + \log(1-D(y))]$

      2. Tries to synthesize fake images that fool the best D:

         $\arg \min_G \max_D \mathbb{E}_{x, y}[\log D(G(x)) + \log(1-D(y))]$

   4. Training Process: 

      1. Sample $x \sim p_{data}, z \sim p_z$
      2. Calc $L_D$ and backward
      3. Calc $L_{G}$ and backward

   5. Example Img2Img Tasks: Labels->Facades, Day->Night, Thermal->RGB, ...

## Lec 17-18: Generative Models

1. Revision of an Early Vision Texture Model:

   1. Select $x \sim p_{data}; z \sim p_z$ (z is usually noise)
   2. Multi-scale filter decomposition (Convolve both images with filter bank)
   3. Match per channel histograms (from noise to data)
   4. Collapse pyramid and repeat

2. Make it better?

   1. Match joint histograms of pairs of filter responses at adjacent spatial locations, orientations, scales, ...
   2. Optimize using repeated projections onto statistical constraint surfaces.

3. Make it more modern: Use CNN to do **texture synthesis**

   1. Previously, we use histograms to describe texture features. Now, we use **Gram Matrices** on **CNN Features** as texture features.

      Define CNN output of some layer as:
      $$
      F_{N\times C} = [f_1, f_2, \dots, f_N]^T
      $$
      We have:
      $$
      G = FF^T = \begin{bmatrix}
      \langle f_1, f_1 \rangle & \cdots & \langle f_1, f_N \rangle \\
      \vdots & & \vdots \\
      \langle f_N, f_1 \rangle & \cdots & \langle f_N, f_N \rangle 
      \end{bmatrix}
      $$
      where 
      $$
      \langle f_i, f_j \rangle = \sum_k F_{ik} F^T_{kj}
      $$
      This describes the **correlation** of image feature $f_i$ and $f_j$, which are both length C (channel) vectors.

   2. Define loss as:
      $$
      L_{\text{style}} = \sum_l \frac{1}{C_l^2 H_l^2 W_l^2} \| G_l(\hat{I}) - G_l(I_{\text{style}}) \|_F^2
      $$

   3. Pipeline for Texture Synthesis:

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202024139897.png)

   4. Remark: The CNN used here is just a pre-trained texture-recognition CNN network, where VGG-16 or VGG-19 nets can be used.

      Basically, select any CNN network that is trained to map from image to label (e.g. "dog") will recognize features totally fine. They are already trained on ImageNet dataset.

4. Use CNN to do **artistic style transfer**

   1. Loss Function Design:

      1. $$
         L_{\text{style}} = \sum_l \frac{1}{C_l^2 H_l^2 W_l^2} \| G_l(\hat{I}) - G_l(I_{\text{style}}) \|_F^2
         $$

      2. $$
         L_{\text{content}} = \frac{1}{2} \sum_{i,j} \left( F_{i,j}^{\text{generated}} - F_{i,j}^{\text{content}} \right)^2
         $$

   2. Pipeline:

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202025144527.png)

5. Diffusion

   1. Training / Inference(Forward):

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202025421867.png" alt="image-20241202025421867" style="zoom:80%;" />

   2. Sampling methods: DDPM vs DDIM

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202030637821.png)

   3. Make sampling faster? Distilation!

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202031116988.png" alt="image-20241202031116988" style="zoom:100%;" />

   4. Editing desired area? Generate a mask that a word corresponds to!

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202031351263.png" alt="image-20241202031351263" style="zoom:100%;" />

6. Common Image Generating Models:

   1. Parti: self-regressive model; generates images block by block
   2. Imagen: Diffusion
   3. Dalle-2: Parti + Imagen

## Lec 19: Sequence Models

1. Shannon, 1948: N-gram model; Compute prob. dist. of each letter given N-1 previous letters (Markov Chain)

2. Video Textures, Sig2000: 

   1. Compute L2 distance $D_{i, j}$ for between all frames

   2. Markov Chain Repr

   3. Transition costs: $C_{i \rightarrow j} = D_{i+1, j}$; Probability Calculated as: $P_{i \rightarrow j} \propto \exp(-C_{i \rightarrow j} / \sigma^2)$

   4. Problem: Preserving Dymanics? Solution: Use previous N frames to calculate cost.

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202162110712.png" alt="image-20241202162110712" style="zoom:70%;" />

   5. Problem: Control video texture speed? Solution: change the weighted sum parameters of previous N costs.

   6. Problem: User control? (e.g. fish chasing mouse pointer)? Solution: add user control term $L = \alpha C + \beta \text{angle}$

      1. Maybe need to precompute future costs for a few angles

3. Efros & Leung Texture Synthesis Algorithm:

   (Bigger window size is better, but requires more params!)

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241202163024888.png" alt="image-20241202163024888" style="zoom:70%;" />

4. Image Analogies Algorithm: Process an image by example (A:A' :: B:B') (Siggraph 2001)

   1. Compare area of pixels (e.g. 10\*10) from img A and B. 
   2. Find the best match, then copy some smaller area of pixels (e.g. 3\*3) from imgA' to imgB'
   3. Remark: Neurips 2022, later method uses VQ-GAN and MAE to finish this

5. Word Embedding (word2Vec, GloVe)...

6. Attention + Prediction: Word sequence tasks

   1. possible explanation: different layers (attention+prediction) works for different functions (syntax, semantics, ...)

7. Similar methods for image generation: treat image as blocks of pixels and generate in order (Parti)

## Lec 20: Single View 3D Vision Geometry

1. Projecting points: use homo coords $(sx, sy, s)$

2. Projecting lines:

   1. A line in the image correspond to a plane of rays through origin.

   2. Computing vanishing points:

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202173331185.png" alt="image-20241202173331185" style="zoom:80%;" />

      Remark1: Any two parallel lines have same vanishing point.

      Remark2: An image may have more than one vanishing point.

   3. The union of all vanishing points is the horizon line (vanishing line)

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202174551595.png" alt="image-20241202174551595" style="zoom:80%;" />

      Different planes define different vanishing lines

      Compute from two sets of parallel lines on the ground plane

      All points at same height as C projects to I

3. 3D from single image

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241202180254612.png" alt="image-20241202180254612" style="zoom:80%;" />

   1. Find world coordinates (X, Y, Z) for a few points.

      1. Define ground plane (Z=0)

         Detecting lines in image? Use **Hough Transform**

      2. Compute points (X,Y,0) on that plane (by homography)

         <img src="/images/CV-Cheatsheet-Finalterm/image-20241202180317320.png" alt="image-20241202180317320" style="zoom:80%;" />

      3. Compute the heights Z of all other points (using perspective clues)

         <img src="/images/CV-Cheatsheet-Finalterm/image-20241202182136063.png" alt="image-20241202182136063" style="zoom:80%;" />

## Lec 21: Multiple View 3D Geometry

1. Disparity Map * Depth Map = Constant.

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241202203703654.png" alt="image-20241202203703654" style="zoom:80%;" />

2. Correspondence problem:

   1. Calculate Epipolar; Match along epipolar line

   2. Effect of window size: want window large enough to have sufficient intensity variation, yet small enough to contain only pixels about the same disparity

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202211455374.png" alt="image-20241202211455374" style="zoom:80%;" />

3. Feature Correspondence Pipeline:

   1. Detect Keypoints; 2. Extract SIFT at each keypoint; 3. Finding correspondences
   2. ALSO: CNN-based stereo matching / depth(disparity) estimation
      1. Feature Extraction; Calculate Cost Volume (pixel i's matching cost at disparity d); Cost Aggregation; Disparity Estimate (simple argmin)

4. Camera Projection Model: $x_{uv} = K[R, t]X_{xyz}$ where $[R, t]$ is w2c matrix

5. Calibrate a Camera? Learning Problem. (use least square; just like solving homography)

   Once we have solved M, decomposite it to K*R using RQ decomposition.

   <img src="/images/CV-Cheatsheet-Finalterm/image-20241202211115754.png" alt="image-20241202211115754" style="zoom:50%;" />

6. Epipolar Geometry:

   1. Introduction: Camera may have rotation, along with translation

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202211542711.png" alt="image-20241202211542711" style="zoom:50%;" />

   2. Definitions

      Baseline: the line connecting the two camera centers

      Epipole: point of intersection of baseline with the image plane

      Epipolar plane: the plane that contains the two camera centers and a 3D point in the world

      Epipolar line: intersection of the epipolar plane with each image plane

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202211629113.png)

      EXAMPLE 1: parallel movement 

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202212231288.png)

      EXAMPLE 2: forward movement

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202212247058.png)

   3. Usage: Stereo image rectification

      Reproject image planes onto a common plane parallel to the line between optical centers

      Then pixel motion is horizontal after transformation

      Two homographies (3x3 transforms), one for each input image reprojection

   4. Calculations: How to express epipolar constraints? (When camera is calibrated)

      Answer: use **Essential Matrix**

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202220501135.png)

      Proof: $X'=RX+T; T \times X' = T \times RX + T \times T = T \times RX; X' \times (T \times X') = 0$

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202220808598.png)

      Properties of $E=T_xR$: 

      - $ Ex' $ is the epipolar line associated with $ x' $ ($ l = E x' $) 
      - $ E^Tx $ is the epipolar line associated with $ x $ ($ l' = E^T x $) 
      - $ E e' = 0 $ and $ E^T e = 0 $ 
      - $ E $ is singular (rank two) 
      - $ E $ has five degrees of freedom (3 for rotation $ R $, 2 for translation $ t $ since it is up to a scale)

   5. Calculations: How to express epipolar constraints? (When camera is **un**-calibrated)

      Answer: **Estimate** the **Fundamental** matrix.

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202221944966.png)

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202221955655.png)

      Estimate F from at least 8 corresponding points

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202222111372.png)

      *: Rank constrain: must do SVD on F and throw out the smallest singular value to enforce rank-2 constraint

## Lec 22: SFM, MVS

1. Problem: unknown 3D Points, Correspondences, Camera Calibration

2. Solution: 

   1. Feature Detection; Matching between each pair using RANSAC; 

      <img src="/images/CV-Cheatsheet-Finalterm/image-20241202225523897.png" alt="image-20241202225523897" style="zoom:70%;" />

   2. Calculate SFM using Bundle Adjustment; Optimized using non-linear least squares (E.g. Levenberg-Marquardt)

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202230604644.png)

      ![](/images/CV-Cheatsheet-Finalterm/image-20241202230613931.png)

   3. Problem: hard to init all cameras; 

      Solution: only start with 1/2 cameras, then grow (kind of online algorithm); "Incremental SFM"

      1. Choose a pair with many matches and as large a baseline as possible
      2. Initialize model with two-frame SFM
      3. While there are connected images remaining, pick one that sees the most existing 3D points; estimate pose; triangulate new points; run bundle adjustment

## Appendix

1. Vector cross product
   $$
   [a_x] = \begin{bmatrix}
   0 & -a_z & a_y \\
   a_z & 0 & -a_x \\
   -a_y & a_x & 0
   \end{bmatrix}
   $$
