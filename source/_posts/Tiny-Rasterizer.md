---
title: My Tiny Rasterizer
mathjax: true
date: 2025-02-18 23:51:25
tags:
- Computer Graphics
- Rasterization
category: UCB-CG-Project
header_image:
abstract: My tiny rasterizer, with super-fast parallel rasterization, antialiasing, trilinear sampling, and elliptical weighted average (EWA) filtering support.
---

## Overview

This is my implementation of a rasterizer, with core features of:

+ **Super-fast rasterization\*** algorithm with line scanning (instead of bounding-box based methods)
+ **Multithread rendering\*** with OpenMP
+ **Antialiasing** support (maximum of MSAAx16 on each pixel)
+ **Perspective Transforms\*** support and correct texture sampling
+ **Trilinear Sampling** support (bilinear interpolation & mipmap on texture)
+ **Elliptical Weighted Average (EWA) Filtering\*** support (on texture sampling)

Terms marked with an asterisk (**\***) indicate my bonus implementations.

> For more detailed mathematical derivation of EWA Filtering,  
> Please refer to my blog: [Mathematical Derivation of EWA Filtering](https://simonxie2004.github.io/2025/02/19/EWA-Filter/)

## Task 1: Rasterizing Triangles

> Keep takeaways: line-scanning method, openMP parallelization.

In this part, we rasterize triangles by **line-scanning** method. This is a method optimized from bounding-box scanning method, which avoids doing line-tests with those pixels that are not in the bounding box. 

### Method Overview

Here is a short pseudocode about how this method works:

1. Sort each vertex such that they satisfy $y_0 < y_1 < y_2$
2. For each **vertical** coordinate $j$ in range $[y_0, y_2]$:
   1. Calculate **left and right boundary**\* as $[x_{\text{left}}, x_{\text{right}}]$
   2. For each **horizontal** coordinate $i$ in range $[x_{\text{left}}, x_{\text{right}}]$: 
      1. Fill color at coordinate $(i, j)$ on screen space.

> *Kindly reminder: You should be <font color=red>**extremely careful**</font> when calculating left and right boundary. The lines may cover half (or even less) of a pixel; However, when doing **MSAA antialiasing**, you still must **calculate the sub-pixel here**. Hence, my recommend is doing two edge interpolations (using both $y$ and $y+1$ instead of solely $y+0.5$), and use the widest horizontal range you can find.

Here is a comparison of **line-scanning** based method and bounding-box based ones.

<table border="1" width="100%">
    <tr>
        <td width="50%"><img src="/images/Tiny-Rasterizer/scanlinerasterizer.png" width="100%"></td>
        <td width="50%"><img src="/images/Tiny-Rasterizer/02-triangle-pixels-bbox.png" width="100%"></td>
    </tr>
    <tr>
        <td align="center"> <b>line-scanning</b> rasterization (faster)</td>
        <td align="center">bounding-box rasterization</td>
    </tr>
</table>
### Method Optimization

Taking intuition from **Bresenham's Line Algorithm**, we can optimize the double for-loop in `rasterize_triangle` function by replacing the `float` variables `i, j` by `int` objects. This will avoid the time-consuming float operations.

Also, this facilitates another optimization, i.e. **parallelized rendering**. To further improve the speed, I additionally added `openmp` support, and parallelized pixel-sampling process.

### Performance Comparison

By adding `clock()` before/after `svg.draw()` call, I recorded the time difference of the rendering process. Here is a performance comparison of some methods: 

<table border="1">
    <tr>
        <th>Method</th>
        <th>B-Box</th>
        <th>Line-Scanning</th>
        <th>Line-Scanning<br>(with OpenMP)</th>
    </tr>
    <tr>
        <td>Render Time (ms)</td>
        <td>80ms</td>
        <td>45ms</td>
        <td><b>18ms</b> (fastest)</td>
    </tr>
</table>

> REMARK: I used the final version with texture to test performance.
>
> Render target: `.\svg\texmap\test1.svg`, same as the result gallery in **Task 5**
>
> Resolution: 800 * 600; Use num_thread: 12
>
> CPU: Intel Core Ultra 9 185H (16 cores (6\*P, 8\*E, 2\*LPE), 22 threads)

### Result Gallery

We can show some examples of rasterization result. As we can see, without antialiasing, the edges of triangles remain jaggy. In the following sections, we will use MSAA to solve this problem.

<table border="1" width="100%">
    <tr>
        <td width="50%"><img src="/images/Tiny-Rasterizer/screenshot_2-17_17-42-1.png" width="100%"></td>
        <td width="50%"><img src="/images/Tiny-Rasterizer/screenshot_2-17_17-42-40.png" width="100%"></td>
    </tr>
    <tr>
        <td align="center">thin triangles</td>
        <td align="center">a cube</td>
    </tr>
</table>
## Task 2: Anti-Aliasing

> Key takeaways: MSAA sampling method, screen buffer data structure.

As seen before, we have a jaggy result from the previous part. Hence, we add MSAA antialiasing in this subpart. 

### Method Overview

MSAA is the simplest method to **smooth edges and reduce visual artifacts**. By **sampling multiple points per pixel and averaging the colors**, we alleviate the jaggies.

<img src="/images/Tiny-Rasterizer/supersampling.jpg" alt="Supersampling." style="zoom: 60%;" />

Here is a pseudocode of my modification to the previous rasterization function. (Grey parts remains same; Blue parts are newly added.)

1. <font color=gray>Sort each vertex such that they satisfy $y_0 < y_1 < y_2$ </font>
2. <font color=gray>For each vertical coordinate $j$ in range $[y_0, y_2]$:</font>
    1. <font color=gray>Calculate left and right boundary as $[x_{\text{left}}, x_{\text{right}}]$</font>
    2. <font color=gray>For each horizontal coordinate $i$ in range $[x_{\text{left}}, x_{\text{right}}]$: </font>
       1. <font color=blue>For each sub-pixel in pixel $(i, j)$</font>
          1. <font color=blue>If sub-pixel $(i + dx, j + dy)$ in triangle:</font>
             1. <font color=blue>Set `screen_buffer[i, j, r]`^ to color `c`</font>
          2. <font color=blue>Else:</font>
             1. <font color=blue>Do nothing*</font>

> ^Note: `r` is the r-th sub-pixel in pixel $(i, j)$, which will be temporarily saved in screen buffer.
>
> *Kindly Reminder: You mustn't do anything sub-pixel $(i+dx, j+dy)$ if it is NOT in the current triangle. Because if it was in another triangle previously (but not the current one), the rendered color will be **overwritten**.

### Key Structure/Function

Also, we must do the following modification to the previous data structure of `screen_buffer` and `resolve_buffer` function. Here are the modifications:

<table border="1">
    <tr>
        <td> </td>
        <td>screen_buffer</td>
        <td>resolve_buffer</td>
    </tr>
    <tr>
        <td>Previously</td>
        <td>Array of W*H</td>
        <td>Array of <b>W*H*S</b> (S=samples per pixel)</td>
    </tr>
    <tr>
        <td>Currently</td>
        <td>Directly render screen_buffer[i, j]</td>
        <td>Render <b>avg(screen_buffer[i, j, :])</b></td>
    </tr>
</table>
### Result Gallery

Here is a table that organizes and compares the rendered images under different MSAA configurations.

<table border="1">
    <tr>
        <th>MSAA x1</th>
        <th>MSAA x4</th>
        <th>MSAA x9</th>
        <th>MSAA x16</th>
    </tr>
    <tr>
        <td><img src="/images/Tiny-Rasterizer/screenshot_2-17_23-34-27.png" width="200"></td>
        <td><img src="/images/Tiny-Rasterizer/screenshot_2-17_23-34-29.png" width="200"></td>
        <td><img src="/images/Tiny-Rasterizer/screenshot_2-17_23-34-30.png" width="200"></td>
        <td><img src="/images/Tiny-Rasterizer/screenshot_2-17_23-34-31.png" width="200"></td>
    </tr>
    <tr>
        <td><img src="/images/Tiny-Rasterizer/screenshot_2-17_23-34-39.png" width="200"></td>
        <td><img src="/images/Tiny-Rasterizer/screenshot_2-17_23-34-40.png" width="200"></td>
        <td><img src="/images/Tiny-Rasterizer/screenshot_2-17_23-34-41.png" width="200"></td>
        <td><img src="/images/Tiny-Rasterizer/screenshot_2-17_23-34-42-173986416037017.png" width="200"></td>
    </tr>
</table>
+ The first row displays the middle section of a red triangle, highlighting the jagged edges and how they are progressively smoothed by increasing levels of MSAA.
+ The second row presents a more extreme case: a zoomed-in view of a very skinny triangle corner. In the first image, some triangles appear disconnected, as the pixels covered by the triangle occupy too little area to be rendered. However, with higher MSAA settings, the true shape of the triangle becomes fully visible.

Here is the most extreme case: an image full of degenerate triangles. My implementation successfully passes this test.

<table border="1" width="100%">
    <tr>
        <th width="50%">MSAA x1</th>
        <th width="50%">MSAA x16</th>
    </tr>
    <tr>
        <td width="50%" align="center">
            <img src="/images/Tiny-Rasterizer/screenshot_2-18_0-35-58.png" alt="z" width="100%">
        </td>
        <td width="50%" align="center">
            <img src="/images/Tiny-Rasterizer/screenshot_2-18_0-38-11.png" alt="screenshot_2-18_0-38-11" width="100%">
        </td>
    </tr>
    <tr>
        <td width="50%" align="center">
            <img src="/images/Tiny-Rasterizer/screenshot_2-18_0-41-24.png" alt="extra-image-1" width="100%">
        </td>
        <td width="50%" align="center">
            <img src="/images/Tiny-Rasterizer/screenshot_2-18_0-41-22.png" alt="extra-image-2" width="100%">
        </td>
    </tr>
</table>


## Task 3: Transforms

> Key takeaways: homogeneous coordinates, transform matrices

In this section, we define the 2D transform matrices to support 2D Transforms. The transforms are done in homogeneous coordinates, mainly because **translation is not a linear operator**, while matrix multiplication is. 

Here we define the 2D Homogeneous coordinates as $[x, y, 1]$. Hence the transforms can be expressed by the following matrices:
$$
\text{Translate:} 
\begin{bmatrix}
1 & 0 & tx \\
0 & 1 & ty \\
0 & 0 & 1
\end{bmatrix}, \quad
\text{Scaling:} 
\begin{bmatrix}
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & 1
\end{bmatrix}, \quad
\text{Rotation:} 
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
$$
Here is `my_robot.svg`, depicting a stick figure raising hand, drawn using 2D transforms.

<img src="/images/Tiny-Rasterizer/screenshot_2-18_0-51-31.png" alt="screenshot_2-18_0-51-31" style="zoom:50%;" />

Also, I have implemented **perspective transforms** by calculating the four-point, 8 DoF matrix, i.e. solving the following 4-point defined ($ABCD \rightarrow A'B'C'D'$) transform:
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
Here is an example result:

<img src="/images/Tiny-Rasterizer/screenshot_2-20_0-26-28.png" alt="screenshot_2-20_0-26-28" style="zoom:70%;" />

## Task 4: Barycentric Coordinates

> Key takeaways: area explanation & distance explanation of barycentric coords.

Now we are adding more colors to a single triangle. Try imagining a triangle with 3 vertices of different colors, what will it look like inside? This is the intuition of barycentric coordinates: interpolating attributes of 3 vertices inside a triangle. This algorithm is quite useful in multiple ways, for example, shading.

### Explanation 1: Proportional Distances

The first explanation comes from the intuition of distance: we have the params defined as follow for interpolation and use the similar construction for other coordinates: 
$$
\alpha = \frac{L_{BC}(x, y)}{L_{BC}(x_A, y_A)}
$$
<img src="/images/Tiny-Rasterizer/image-20250218011200298.png" alt="image-20250218011200298" style="zoom: 53%;" />

### Explanation 2: Area Interpolation

The second explanation comes from area: we have the params defined as proportions of area (and use similar construction for other coordinates):
$$
\alpha = \frac{A_A}{A_A + A_B + A_C}
$$
<img src="/images/Tiny-Rasterizer/image-20250218011411655.png" alt="image-20250218011411655" style="zoom:53%;" />

### Result Gallery

With barycentric coordinates, we can interpolate any point inside a triangle, even we only have the vertex colors! Here is a example result combined of multiple triangles:

<img src="/images/Tiny-Rasterizer/screenshot_2-18_1-18-43.png" alt="screenshot_2-18_1-18-43" style="zoom:60%;" />

## Task 5: Pixel Sampling

> Key takeaways: nearest / bilinear / **elliptical weighted average** sampling

In this section, I implement three different sampling methods, i.e. nearest, bilinear and elliptical weighted average (EWA) filtering (which is one of the **anisotropic** sampling methods), and compare their performance. 

### Method Overview

Mainly speaking, sampling is the process of **back-projecting** $(x, y)$ coordinate to $(u, v)$ space, and selecting a texel to represent pixel $(x, y)$'s color. However, the back-projected coordinates aren't always integers, which aren't given any color information (since texels only provide color information on integer grids). Hence we need **sampling** methods. Here are the methods that are implemented:

1. **Nearest Sampling** selects the closest texel to the given sampling point without any interpolation, which is fast and efficient but can result in blocky, pixelated visuals.
2. **Bilinear sampling** improves image quality by using the weighted average between the four nearest texels surrounding the sampling point. However, it can still introduce blurriness when scaling textures significantly.
3. **EWA Filtering Sampling** (EWA) filtering is a high-quality texture sampling method. It considers multiple texels in an elliptical region, weighting them based on their distance to the sampling point using a **gaussian kernel**. This method reduces aliasing and preserves fine details, but it is computationally expensive compared to nearest and bilinear sampling.

Here is a comparison of the three methods:

<table border="1" width="100%">
    <tr>
        <td width="33%" align="center">
            <img src="/images/Tiny-Rasterizer/image-20250218043447029.png" alt="Nearest Sampling" width="100%">
        </td>
        <td width="33%" align="center">
            <img src="/images/Tiny-Rasterizer/image-20250218043411733.png" alt="Bilinear Sampling" width="100%">
        </td>
        <td width="33%" align="center">
            <img src="/images/Tiny-Rasterizer/image-20250218043540126.png" alt="EWA Filtering" width="100%">
        </td>
    </tr>
    <tr>
        <td align="center"><strong>Nearest Sampling</strong></td>
        <td align="center"><strong>Bilinear Sampling</strong></td>
        <td align="center"><strong>EWA Filtering</strong></td>
    </tr>
</table>

### Perspective Transforms: An Exception

When applying barycentric coordinates in rasterization, we typically assume that interpolation in screen space is equivalent to interpolation in the original 2D space. However, when a **perspective transform** is applied, this assumption breaks down because perspective projection **distorts distances non-linearly**. This means that linearly interpolating attributes (e.g., color, texture coordinates) in screen space **does not match** the correct interpolation in world space.

But there is still solution: we can first back-project the triangle vertices to the original space. When sampling screen-pixel $(i, j)$, we can also do the same back-project and then interpolate. Here is the fixed pseudocode:

1. Back-Project triangle vertices $A, B, C$ back to $A', B', C'$
2. <font color=grey>... inside rasterization loop, sampling at $(i, j)$</font>
   1. Back-Project screen-space pixel $(i, j)$ back to $(i', j')$
   2. Calculate barycentric interpolation using $A', B', C'$, $(i', j')$
   3. Pass $\alpha, \beta, \gamma$ to texture sampling function...

Here is a comparison of **wrong** implementation and **correct** implementation:

<table border="1" width="100%">
    <tr>
        <td width="50%">
            <img src="/images/Tiny-Rasterizer/screenshot_2-20_0-34-10.png" width="100%">
        </td>
        <td width="50%">
            <img src="/images/Tiny-Rasterizer/screenshot_2-20_0-34-43.png" width="100%">
        </td>
    </tr>
</table>

### Result Gallery

The first two standard methods' results are as follows:

<table border="1" width="100%">
    <tr>
        <td width="20%"></td>
        <td width="40%" style="text-align: center;">MSAA x1</td>
        <td width="40%" style="text-align: center;">MSAA x16</td>
    </tr>
    <tr>
        <td width="20%" style="text-align: center;">Nearest Sampling</td>
        <td width="40%"><img src="/images/Tiny-Rasterizer/screenshot_2-18_4-50-41-173988356465430.png" alt="Nearest x1" width="100%"></td>
        <td width="40%"><img src="/images/Tiny-Rasterizer/screenshot_2-18_4-50-45-173988357302932.png" alt="Nearest x16" width="100%"></td>
    </tr>
    <tr>
        <td width="20%" style="text-align: center;">Bilinear Sampling</td>
        <td width="40%"><img src="/images/Tiny-Rasterizer/screenshot_2-18_4-50-49-173988357805934.png" alt="Bilinear x1" width="100%"></td>
        <td width="40%"><img src="/images/Tiny-Rasterizer/screenshot_2-18_4-50-52-173988358271836.png" alt="Bilinear x16" width="100%"></td>
    </tr>
</table>

The differences are obvious. In the cases that the texels are quite sparse or sharp (for example, the **meridians** in the map), nearest sampling will create a lot of disconnected subparts which looks quite like jaggies. However, with bilinear sampling, the disconnected parts appear connected again.

Also, the render result of anisotropic filtering (EWA) method is provided as follows. Pay attention to the **distant part** of the ground plane and the **top surface** of the block. You can zoom-in by clicking on the image.

<table width="100%" border="1">
    <tr>
        <td width="50%"><img src="/images/EWA-Filter/bilinear+bilinear+msaa1x.png" alt="Bilinear + Bilinear + MSAA1x" width="100%"></td>
        <td width="50%"><img src="/images/EWA-Filter/bilinear+ewa+msaa1x.png" alt="Bilinear + EWA + MSAA1x" width="100%"></td>
    </tr>
    <tr>
        <td width="50%" align="center">Bilinear Pixel + Bilinear Level + MSAA1x</td>
        <td width="50%" align="center">Bilinear Pixel + <b>EWA Filtering</b> + MSAA1x</td>
    </tr>
</table>

> For more detailed mathematical derivation of EWA Filtering,  
> Please refer to my blog: [Mathematical Derivation of EWA Filtering](https://simonxie2004.github.io/2025/02/19/EWA-Filter/)

## Task 6: Level Sampling

> Key takeaways: calculating $\frac{\partial dx}{\partial du}$, mip level L, mipmaps

The concept of level sampling naturally arises from the previous problem. When sampling from texture space, a single pixel may cover a large area, making it difficult to accurately represent the texture. Simply sampling the nearest 1–4 pixels can lead to unrepresentative results, while manually computing the area sum is inefficient. To address this issue, we use level sampling techniques, such as mipmapping, which enhance both accuracy and efficiency.

![image-20250218194013574](/images/Tiny-Rasterizer/image-20250218194013574.png)

> Example: different pixels have different texture space footprint

### Mipmap Overview

Here we implement **mipmap**, which is a list of downsampled texture, each at progressively lower resolutions. When rendering, the appropriate mipmap level is chosen based on the texture-space pixel size, ensuring that a pixel samples from the most suitable resolution. This reduces aliasing artifacts and improves performance by avoiding unnecessary high-resolution texel averaging.

![img](/images/Tiny-Rasterizer/mipmaps.png)

Here is an image intuitive of $L$ and partial derivatives. We inspire from it to learn how to choose the corresponding mip level to sample texture from.

![image-20250218200526288](/images/Tiny-Rasterizer/image-20250218200526288.png)Therefore, we can calculate the level $L$​ as follows:
$$
L = \max \left( \sqrt{\left(\frac{du}{dx}\right)^2 + \left(\frac{dv}{dx}\right)^2}, \sqrt{\left(\frac{du}{dy}\right)^2 + \left(\frac{dv}{dy}\right)^2} \right)
$$
where $\frac{du}{dx}$ is calculated as follows:

1. Calculate the uv barycentric coordinates of $(x, y)$ $(x+1, y)$ $(x, y+1)$
2. Calculate the corresponding uv coordinates $(u_0, v_0)$ $(u_1, v_1)$ $(u_2, v_2)$
3. Hence $\frac{du}{dx} = u_1 - u_0$, and the rest are the same.

### Level Blending Overview

By blending between mip levels (trilinear filtering), we can achieve smoother transitions and further enhance visual quality. If a level is calculated as a float, we use linear-interpolation, which samples from both levels and do a weighted average. This reduces the artifacts created by sudden change of level of detail.

### Result Gallery

<table width="100%" border="1">
  <tr>
    <th width="16%">Level/Pixel <br> Sampling</th>
    <th width="28%">Nearest</th>
    <th width="28%">Bilinear</th>
    <th width="28%">EWA</th>
  </tr>
  <tr>
    <th>L_zero</th>
    <td><img src="/images/Tiny-Rasterizer/screenshot_2-18_23-39-4.png" width="100%"></td>
    <td><img src="/images/Tiny-Rasterizer/screenshot_2-18_23-39-7.png" width="100%"></td>
    <td><img src="/images/Tiny-Rasterizer/screenshot_2-18_23-39-15.png" width="100%"></td>
  </tr>
  <tr>
    <th>L_nearest</th>
    <td><img src="/images/Tiny-Rasterizer/screenshot_2-18_23-39-27.png" width="100%"></td>
    <td><img src="/images/Tiny-Rasterizer/screenshot_2-18_23-39-30.png" width="100%"></td>
    <td><img src="/images/Tiny-Rasterizer/screenshot_2-18_23-39-35.png" width="100%"></td>
  </tr>
  <tr>
    <th>L_linear</th>
    <td><img src="/images/Tiny-Rasterizer/screenshot_2-18_23-39-42.png" width="100%"></td>
    <td><img src="/images/Tiny-Rasterizer/screenshot_2-18_23-39-44.png" width="100%"></td>
    <td><img src="/images/Tiny-Rasterizer/screenshot_2-18_23-39-50.png" width="100%"></td>
  </tr>
</table>
## Final Discussion

1. **Speed**. Better methods are more time-consuming. Sorted by render time:
   1. Pixel sampling: EWA >> Bilinear > Nearest
   2. Level sampling: L_bilinear >> L_nearest > L_zero
   3. Antialiasing: MSAAxN cost N times time.
2. **Memory**.
   1. Pixel sampling doesn't cost more memory.
   2. Level sampling: Mipmap costs 2x texture space memory.
   3. Antialiasing requires $r$ times memory buffer, but can be done in-place.




