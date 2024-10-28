---
title: Automatic Image Panorama
mathjax: true
date: 2024-10-27 20:22:47
tags:
- Computer Vision
- Image Processing
category: UCB-CV-Project
header_image: /images/Automatic-Image-Panorama/teaser_final.png
abstract: UC-Berkeley 24FA CV Project 4a & 4b - Automatic Image Panorama.
---

> UC-Berkeley 24FA CV Project 4a & 4b: 
>
> Creating panorama images by correspondence detecting, projective warping, resampling and compositing them together.  
> A simpler reproduction of [Multi-Image Matching using Multi-Scale Oriented Patches](https://ieeexplore.ieee.org/document/1467310)

## Project 4a: Manual Image Panorama

> Creating panoramas by manually registering corresponding points

### Shoot and Digitize pictures

1. First, we retrieve some images surrounding memorial glade @ Berkeley.

   These images have same center of projection (COP) and different looking directions.

   <table>
      <tr>
        <td><img src="/images/Automatic-Image-Panorama/00.jpg" alt="00"></td>
        <td><img src="/images/Automatic-Image-Panorama/01.jpg" alt="01"></td>
        <td><img src="/images/Automatic-Image-Panorama/02.jpg" alt="02"></td>
      </tr>
    </table>

2. Then we define the following corresponding points manually:

   <img src="/images/Automatic-Image-Panorama/corr1.png" alt="corr1" style="zoom:80%;" />

   <img src="/images/Automatic-Image-Panorama/corr2.png" alt="corr2" style="zoom:80%;" />

### Recover Homographies

1. The transform between images is basically a projection transform with 8 degree of freedoms. This can be written as:
   $$
   H \mathbf{p} = \mathbf{p}'
   $$
   where:
   $$
   H = 
   \begin{pmatrix}
   a & b & c \\
   d & e & f \\
   g & h & 1
   \end{pmatrix}
   $$

2. Expanding it out would get:
   $$
   \begin{aligned}
   w'x' &= a x + b y + c \\
   w'y' &= d x + e y + f \\
   w' &= g x + h y + 1
   \end{aligned}
   $$

3. Further expansion (into eight separate variables and a linear system) will result:
   $$
   \begin{pmatrix}
   x_1 & y_1 & 1 & 0 & 0 & 0 & -x_1 x_1' & -y_1 x_1' \\
   0 & 0 & 0 & x_1 & y_1 & 1 & -x_1 y_1' & -y_1 y_1' \\
   x_2 & y_2 & 1 & 0 & 0 & 0 & -x_2 x_2' & -y_2 x_2' \\
   0 & 0 & 0 & x_2 & y_2 & 1 & -x_2 y_2' & -y_2 y_2' \\
   \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
   x_N & y_N & 1 & 0 & 0 & 0 & -x_N x_N' & -y_N x_N' \\
   0 & 0 & 0 & x_N & y_N & 1 & -x_N y_N' & -y_N y_N'
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

4. Thus, we have least square method to solve this over-determined equation.

   `h = np.linalg.inv(A.T @ A) @ (A.T @ b)` OR `h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)`

### Warp the Images

1. With the projection matrix, we may warp any image 1 to image 2. We implement this by: 
   
   1. First, project the four corners of image 1 to image 2, defining an `area` that img1 will be projected onto.
   2. Second, rasterize the `area`, covering all pixels that img1 will be projected onto.
   3. Third, inversely project all pixels in `area` back to image 1 and sample them bilinearly. This will give us the projected image.

   > Remark: using `scipy.interpolate.griddata` is too slow here since it has to query the whole point set. Hence we can implement the bilinear-interpolation and vectorize it using numpy operations ourselves.

2. Here is our example result:

   ![](/images/Automatic-Image-Panorama/partialFinal.png)

### Image Rectification

1. According to project requirements, we should perform "rectification" on image to check if the projection works correctly.

2. Here is an example:

   ![](/images/Automatic-Image-Panorama/image-20241020012710902.png)

   > Here we require the ipad in the scene to be rectangular (and hence defined corresponding points on four corners.)

3. Another example:

   ![](/images/Automatic-Image-Panorama/image-20241020012716590.png)

   

### Image Blending

1. To avoid weird edge artifacts (usually caused by simply covering one image by another), 

   we develop a method to blend images together with an seamless transition.

2. First, we generate a mask as follows.

   <img src="/images/Automatic-Image-Panorama/image-20241026234849710.png" alt="image-20241026234849710" style="zoom:67%;" />

3. Second, we blend images together according to the mask's values.

   For pixels that only the middle image cover, we use its original value.

   For pixels that are covered by multiple images, set it to `mask * middle_value + (1-mask) * avg(other_values)`

4. Here is an example result:

   ![](/images/Automatic-Image-Panorama/final.png)

5. Ablation study:

   ![](/images/Automatic-Image-Panorama/image-20241026235729895.png)

   ![](/images/Automatic-Image-Panorama/image-20241027000336692.png)

   > Up: w/o blending; Down: with blending

## Project 4a: Result Gallery

### Bancoft St.

<table>
   <tr>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027172653092.png" alt="00"></td>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027172657344.png" alt="01"></td>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027172702010.png" alt="02"></td>
   </tr>
 </table>

![](/images/Automatic-Image-Panorama/image-20241020012614510.png)

### Oxford St.

<table>
   <tr>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027172822261.png" alt="00"></td>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027172901343.png" alt="01"></td>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027172905642.png" alt="02"></td>
   </tr>
 </table>

![](/images/Automatic-Image-Panorama/image-20241020012625562.png)

### My Home

<table>
   <tr>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027173000497.png" alt="00"></td>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027173138528.png" alt="01"></td>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027173154504.png" alt="02"></td>
   </tr>
 </table>

![](/images/Automatic-Image-Panorama/image-20241027000930754.png)

## Project 4b: Automatic Image Panorama

> Here we derive a method to detect corresponding points automatically.
>
> Thus, the image panorama process will be completely automatic, without manually registering points.

### Harris Corner Detector

1. We can implement Harris corner detector, a classic method to find corners in the image.

2. The basic idea of Harris corner detector is: slide a window on the image, and sense the change of pixel values.

   <img src="/images/Automatic-Image-Panorama/image-20241027154454534.png" alt="image-20241027154454534" style="zoom:70%;" />

3. First, we define the change in appearance of window W for the shift $[u, v]$ as follows:
   $$
   E(u, v) = \sum_{(x, y) \in W}{I(x + u, y + v) - I(x, y)}
   $$
   <img src="/images/Automatic-Image-Panorama/image-20241027154853051.png" alt="image-20241027154853051" style="zoom: 70%;" />

4. Then, we use a First-order Taylor approximation for small motions $[u, v]$:
   $$
   \begin{aligned}
   I(x+u, y+v) &= I(x, y) + I_x u + I_y v + \text{higher order terms} \\
   &\approx I(x, y) + I_x u + I_y v \\
   &= I(x, y) + \begin{bmatrix} I_x & I_y \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix}
   \end{aligned}
   $$
   Plugging this into $E(u, v)$, we would get:
   $$
   \begin{aligned}
   E(u, v) &= \sum_{(x, y) \in W} \left[I(x+u, y+v) - I(x, y)\right]^2 \\
   &\approx \sum_{(x, y) \in W} \left[I(x, y) + \begin{bmatrix} I_x & I_y \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} - I(x, y)\right]^2 \\
   &= \sum_{(x, y) \in W} \left(\begin{bmatrix} I_x & I_y \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix}\right)^2 \\
   &= \sum_{(x, y) \in W} \begin{bmatrix} u & v \end{bmatrix} \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix}
   \end{aligned}
   $$

5. This gives us the second moment matrix $M$, which is a approximate of local change on images.
   $$
   M = \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
   $$

    <img src="/images/Automatic-Image-Panorama/Quadratic.png" alt="image-20241027154853051" style="zoom: 90%;" />

6. Therefore, we may use this quadratic form's eigenvalues and eigenvectors to describe a rotation-invariant corner strength.

   ![](/images/Automatic-Image-Panorama/Eigenvalues.png)

   Here, we calculate this function value as "corner strength":
   $$
   R = \det(M) - k * \text{tr}(M)^2
   $$
   where $\det(M) = \lambda_1 * \lambda_2$, $\text{tr}(M) = \lambda_1 + \lambda_2$.

   Note that $k$ is an empirical value, usually between $[0.04, 0.06]$. If this value is bigger, less corners are detected, and vice versa. 

7. Summary:
   1. Compute Gaussian derivatives at each pixel
   2. Compute second moment matrix M in a Gaussian
   window around each pixel
   3. Compute corner response function R & Threshold R

8. Example Results:

  <table>
     <tr>
       <td><img src="/images/Automatic-Image-Panorama/image-20241027161702745.png" alt="00"></td>
       <td><img src="/images/Automatic-Image-Panorama/image-20241027161709931.png" alt="01"></td>
       <td><img src="/images/Automatic-Image-Panorama/image-20241027161820947.png" alt="02"></td>
     </tr>
   </table>

### Adaptive Non-Maximal Suppression

1. Since the computational cost of matching is superlinear in the number of interest points, it is desirable to restrict the number of interest points extracted from each image. Also, we wish the interest points are spatially well-distributed. Hence, we implement ANMS (Adaptive Non-Maximal Suppression) to prune some points.

2. Method: Interest points are suppressed based on the corner strength $R$, and only those that are a maximum in a neighborhood of radius $r$ pixels are retained.

3. In practice we robustify the non-maximal suppression by requiring that a neighbor has a sufficiently larger strength. Thus the minimum suppression radius $r_i$ is given by:
   $$
   r_i = \min_{j} |x_i - x_j|, \text{s.t.} f(x_i) < c_{\text{robust}}f(x_j), X_J \in I
   $$
   where where $x_i$ is a 2D interest point image location, and $I$ is the set of all interest point location. 

   Note that $c_j$ is an empirical value, usually taking $0.9$.

4. Example results:

   <table>
      <tr>
        <td><img src="/images/Automatic-Image-Panorama/image-20241027161702745.png" alt="00"></td>
        <td><img src="/images/Automatic-Image-Panorama/pruned.png" alt="01"></td>
      </tr>
    </table>

   > Left: w/o pruning; Right: with ANMS pruning

### Feature Descriptor Extraction

1. For each filtered interest point, we extract a feature descriptor for it. This is to facilitate our matching process, providing more  information instead of a single pixel R value.

2. In this project, we simply extract a 8\*8 axis-aligned box, down-sampled from a 40\*40 box, centered by any interest point.

3. Here are some example results:

   <table style="width: 80%; margin: auto;">
      <tr>
         <td style="width: 50%; text-align: right;">
            <img src="/images/Automatic-Image-Panorama/image-20241027164051649.png" alt="00" style="height: 300px; object-fit: contain;">
         </td>
         <td style="width: 50%; text-align: left;">
            <img src="/images/Automatic-Image-Panorama/image-20241027164446481.png" alt="01" style="height: 300px; object-fit: contain;">
         </td>
      </tr>
   </table>

### Feature Matching

1. Here, we implement an index to describe matching error, and then threshold on it to filter correctly matched points.  
   To be specific, the ratio of the closest to the second closest (1NN/2NN) is implemented.  
   This value is thresholded by an empirical value 0.67 according to the following graph.

   <img src="/images/Automatic-Image-Panorama/image-20241027165931828.png" alt="image-20241027165931828" style="zoom: 90%;" />

2. Here is an example after pruning points.

   <table>
      <tr>
        <td><img src="/images/Automatic-Image-Panorama/image-20241027170154376.png" alt="00"></td>
        <td><img src="/images/Automatic-Image-Panorama/image-20241027170159178.png" alt="01"></td>
      </tr>
    </table>

### RANSAC Homography Estimation

1. As can be read from the image, there are still some outliers that are incorrectly matched. Hence we use RANSAC (RaNdom SAmple Consensus) to calculate a robust transform matrix.

2. RANSAC basically iterates a lot of times, each time choosing some points from the interest point list and calculate a homography matrix. Then, the algorithm checks if the points projected by the homography matrix are correct (meaning they are close to some other point in the plane that the image is projected to.)

3. Here is an example after calculating a robust transform matrix by RANSAC:

   <table>
      <tr>
        <td><img src="/images/Automatic-Image-Panorama/image-20241027170552202.png" alt="00"></td>
        <td><img src="/images/Automatic-Image-Panorama/image-20241027170556599.png" alt="01"></td>
      </tr>
    </table>

4. With the homography matrix, we can repeat the steps in project 4a to calculate a completely automatic image panorama!

### Result Comparison

<table>
   <tr>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027201226861.png" alt="00"></td>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027200628276.png" alt="01"></td>
   </tr>
 </table>

<table>
   <tr>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027200915214.png" alt="00"></td>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027200713242.png" alt="01"></td>
   </tr>
 </table>

<table>
   <tr>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027200944123.png" alt="00"></td>
     <td><img src="/images/Automatic-Image-Panorama/image-20241027200752485.png" alt="01"></td>
   </tr>
 </table>

> Left: Automatic correspondence detection panorama; Right: Manual registering points panorama
>
> Remark: Due to some errors in manually registering points, the last image in the right has been incorrect; but the automatic panorama's results are correct.

## Project 4b: Result Gallery

### Sather Gate

![](/images/Automatic-Image-Panorama/sather1.png)

### Somewhere near RSF

![](/images/Automatic-Image-Panorama/lowersprout.png)

### Lower plaza

![](/images/Automatic-Image-Panorama/lowerplaza.png)

## Summaries and Takeaways

1. From what HAS been implemented:
   1. Start from simple ideas. Corner detection is a hard job, but detecting the change in a small sliding window is not.
   2. Do it inversely. Direct projection makes it hard to grid points, but inverse-sampling makes it much simpler and parallelizable.
2. From what has NOT been implemented:
   1. Make it multi-resolution. If you can'd detect corners in multiple scales, do it multiple times with different window sizes.
   2. Use transform-invariant descriptors. For example, eigenvalues. Or, make the corner descriptor box aligned to corner direction, instead of the axes.
