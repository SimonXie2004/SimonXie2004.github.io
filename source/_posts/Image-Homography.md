---
title: Image-Homography
mathjax: true
date: 2024-10-19 23:18:38
tags:
- Computer Vision
- Image Processing
category: UCB-CV-Project
header_image: /images/Image-Homography/teaser_final.png
abstract: UC-Berkeley 24FA CV Project 4a - Morphing Images to create Homography.
---

# IMAGE WARPING and MOSAICING

> Creating panorama images by registering, projective warping, resampling and compositing them.

![](/images/Image-Homography/final.png)

## Shoot and Digitize pictures

1. First, we retrieve some images surrounding memorial glade @ Berkeley.

   These images have same center of projection (COP) and different looking directions.

   <table>
      <tr>
        <td><img src="/images/Image-Homography/00.jpg" alt="00"></td>
        <td><img src="/images/Image-Homography/01.jpg" alt="01"></td>
        <td><img src="/images/Image-Homography/02.jpg" alt="02"></td>
      </tr>
    </table>

2. Then we define the following corresponding points manually:

   ![](/images/Image-Homography/corr1.png)

   ![](/images/Image-Homography/corr2.png)

## Recover Homographies

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

   Both `np.linalg.inv(A.T @ A) @ (A.T @ b)` and `np.linalg.lstsq(A, b)` works.

## Warp the Images

1. With the projection matrix, we may warp any img1 to img2. We implement this by: 

   1. First, project all grid points to img2's plane.
   2. Second, reshape image into `[-1, 3]` as griddata.
   3. Third, we use `scipyl.interpolate,griddata` to sample all points on img2's plane with img1's (projected) pixels

2. Here is our example result:

   ![](/images/Image-Homography/image-20241019223834635.png)

## Image Blending

1. In this project, we implement a simple method to average images.

   For any area with $N$ images overlapping, we use factor $\frac{1}{N}$ to average them.

   For example, if there is two projected images, they are averaged by $\frac{1}{2}$.

2. For the area that overlaps, we use this mask to blend to image. 

   For those parts that don't overlap, we simply use the original pixel value from the projected images.

3. Here is an example:

   ![](/images/Image-Homography/partialFinal.png)

   ![](/images/Image-Homography/final.png)

4. More results:

   ![](/images/Image-Homography/image-20241020012614510.png)

   ![](/images/Image-Homography/image-20241020012625562.png)

## Image Rectification

1. By the way, as long as we can warp the images, we may perform "rectification" on image.

2. We can do this by defining a source area with four points, then define a target area and finally transform from source to target.

3. Here is an example:

   The right image is the source. The left image is the projected image.

   Here we require the ipad in the scene to be rectangular (and hence defined corresponding points on four corners.)

   ![](/images/Image-Homography/image-20241020012710902.png)

   ![](/images/Image-Homography/image-20241020012716590.png)
