---
title: Face-Morphing
mathjax: true
date: 2024-10-07 03:56:06
tags:
- Computer Vision
- Image Processing
category: UCB-CV-Project
header_image: /images/Face-Morphing/teaser.png
abstract: UC-Berkeley 24FA CV Project 3 - Face Morphing, Human Faces Averaging, and Face-PCAs. 
---


# Face Morphing

## Overview

In this assignment we will: 

1. Produce a "morph" animation of one's face into someone else's face
2. Compute the mean of a population of faces 
3. Extrapolate from a population mean to create a caricature of oneself.

A morph is a simultaneous warp of the image shape and a cross-dissolve of the image colors. The cross-dissolve is the easy part; controlling and doing the warp is the hard part. The warp is controlled by a human-defined correspondence between the two pictures. The correspondence should map eyes to eyes, mouth to mouth, chin to chin, ears to ears, etc., to get the smoothest transformations possible.

## Part I: Defining Correspondences

1. Use this [tool](https://cal-cs180.github.io/fa23/hw/proj3/tool.html) to define correspondences

2. Implement the `delaunay` triangulation algorithm.

   *Remark: two images must have same Delaunay triangulation (because the triangles must be one-to-one correspondent for two images), hence `delaunay` is only called once!

   <img src="/images/Face-Morphing/image-20241007022157062.png" alt="image-20241007022157062" style="zoom: 67%;" />

   > Creating Delaunay Triangulation. Image cited from CS180, UC Berkeley, Alexei Efros.

3. Here is an example of results

![](/images/Face-Morphing/image-20241007014432865.png)

<table style="max-width: 100%; table-layout: fixed; width: 100%;">
   <tr>
      <td>
         <figure>
         <img src="/images/Face-Morphing/image-20241007014448332.png" alt="image1.jpg" width="300"/>
         </figure>
      </td>
      <td>
         <figure>
         <img src="/images/Face-Morphing/image-20241007014458290.png" alt="image2.jpg" width="300"/>
         </figure>
      </td>
   </tr>
</table>

## Part II: Compute the Mid-Way Face

1. First, we define `computeAffine(tri1, tri2)`, which calculates the affine transform between two triangles.

   This is equivalent to solving this matrix transform (under homogeneous coordinate system):
   $$
   \begin{pmatrix} a & b & tx \\ c & d & ty \\ 0 & 0 & 1 \end{pmatrix} \cdot \begin{pmatrix} x_1 & x_2 & x_3 \\ y_1 & y_2 & y_3 \\ 1 & 1 & 1 \end{pmatrix} = \begin{pmatrix} x_1' & x_2' & x_3' \\ y_1' & y_2' & y_3' \\ 1 & 1 & 1 \end{pmatrix}
   $$

2. Then, compute the mid-way face! This would involve: 

   1. Computing the average shape (a.k.a the average of each keypoint location in the two faces)

   2. Warping both faces into that shape (a.k.a apply affine transforms separately)

      *Remark: This will involve rasterization of triangles (so as to determine the pixels inside different triangles, and transform them with different affine matrices.)

      *Remark: Suppose we have image A and C and we want to create image B. We can warp B to A and B to C separately, so we can sample the color by bilinear interpolation.

   3. Averaging the colors together.

      <img src="/images/Face-Morphing/image-20241007015107083.png" alt="image-20241007015107083" style="zoom:60%;" />

      <img src="/images/Face-Morphing/image-20241007015117556.png" alt="image-20241007015117556" style="zoom:60%;" />

      > Using inverse transform and bilinear interpolation to morph images. 
      >
      > Image cited from CS180, UC Berkeley, Alexei Efros.

3. Here is an example result:

   ![](/images/Face-Morphing/image-20241007014855159.png)

## Part III: Create Morph Sequence

1. As long as we can morph the mid-way image, we can morph any sequence just like linear-interpolation!

   The result is given by `avgColor = c1*t + c2*(1-t)`, and we warp the image into `avgPts = pts1*t + pts2*(1-t)`

2. Here is an example result:

   ![](/images/Face-Morphing/image-20241007022258722.png)

3. To make the result more fluent, we can implement a sigmoid function to make the animation non-linear.

4. The final result is given by:

   ![](/images/Face-Morphing/me2dad.gif)

## Part IV: The "Mean face" of a population

1. In this part, we create a population's mean face!

2. Here, we choose the dataset [FEI Database](https://fei.edu.br/~cet/facedatabase.html) and [IMM  Database](https://web.archive.org/web/20210305094647/http://www2.imm.dtu.dk/~aam/datasets/datasets.html), which include some subpopulation's faces and their annotated points.

   1. FEI Database is a dataset on Brazilians. It includes 100 images for male, female, male_smiling, female_smiling respectively.
   2. IMM Database is a dataset on Danish. For each people, six images with either different poses or emotion are taken.

   <img src="/images/Face-Morphing/landmark_46points.jpeg" alt="img" style="zoom:80%;" />

   > An example image visualized with annotated points from FEI dataset

3. We compute mean face by an approach similar to face morphing. Simply calculating mean `pts`, morph each face into that `pts` and average them will produce an awesome result!

4. For FEI Dataset, here is the average face for all male/female.

   ![](/images/Face-Morphing/FEI_mean.png)

5. For IMM Dataset, here is the average face for all male/female.

   ![](/images/Face-Morphing/IMM_mean_male.png)

   ![](/images/Face-Morphing/IMM_mean_female.png)

6. Also, a few interesting experiments can be done.

   1. We can morph some images in the dataset to the average face on that dataset, and see what happens.

      ![](/images/Face-Morphing/image-20241007032802927.png)

      ![](/images/Face-Morphing/image-20241007032810148.png)

      ![](/images/Face-Morphing/image-20241007032825094.png)

   2. We can morph a picture of me to one dataset, or morph that dataset's average face to mine.

      ![](/images/Face-Morphing/image-20241007032927366.png)

      ![](/images/Face-Morphing/image-20241007032932179.png)

## Part V. Caricatures: Extrapolating from the mean

This part produce a caricature of my face by extrapolating from the population mean calculated in the last step.

1. Suppose we want to make it more "Danish"  or farther from "Danish" (i.e. we are using the IMM mean face above.)

2. The results are given by:

   ![](/images/Face-Morphing/image-20241007033131330.png)

   >  The left-most and right-most images are the results of extrapolation, while other images are results of interpolation.

## Part VI: Bells and Whistles - PCA Face

1. This part is given by:

   1. Calculate a PCA basis on dataset
   2. Visualize components
   3. Create better caricature on PCA method

2. The PCA result on `FEI_female1` is given as follows (we take the first 10 components):

   ![](/images/Face-Morphing/image-20241007033544475.png)

   ![](/images/Face-Morphing/image-20241007033549574.png)

3. Suppose we want to make my face more "feminine". We can implement that based on enhancing PCA Component weight.

   The result is shown as follows:

   ![](/images/Face-Morphing/image-20241007033715996.png)
