---
title: ClothSim
mathjax: true
date: 2025-04-08 16:54:28
tags:
- Computer Graphics
- Physics Simulation
- GLSL Shader
category: UCB-CG-Project
header_image:
abstract: Imimplementation of a physically-based cloth simulator, with numerical integration, collision handling, spatial hashing, and some fancy shaders.
---

## Overview

This is an implementation of a physically-based cloth simulator, with core features of:

+ Mass-and-Spring Based Cloth Definition
+ Simulation via Numerical Integration
+ Collision with Objects
+ Self-Collision through Spatial Hashmap
+ Shaders (Phong, Bump, Displacement, Environment, ...)

<img src="/images/ClothSim/image-20250408010521298.png" alt="image-20250408010521298" style="zoom:67%;" />

## Part 1: Masses and Springs

### Definition

+ We define a cloth as a **Mass-and-Spring** system, with the following constraints:

  + **Structural** Spring: Between any **adjacent** point masses.

  + **Shearing** Spring: Between any **diagonally adjacent** point masses.

  + **Bending** Spring: Between any point masses that are **two steps apart**.

    <img src="/images/ClothSim/image-20250408022250590.png" alt="image-20250408022250590" style="zoom:67%;" />

### Result Gallery

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td><img src="/images/ClothSim/image-20250407220733279.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407220905599.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407220750779.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407220802973.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td style="text-align:center;">All constraints</td>
    <td style="text-align:center;">Closer lookup of <br> all constraints</td>
    <td style="text-align:center;">Bending and Structural constraints</td>
    <td style="text-align:center;">Shearing constraints</td>
  </tr>
</table> 

> Screenshots of scene/pinned2.json, showing the constraints.

## Part 2: Verlet Integration

### Algorithm

The Verlet integration combines from two parts:

First, update all point masses' forces. Algorithm:

1. For all `spring` in `springs`:
   1. Calculate force by Hooke's law.
   2. Apply the force to the connected point masses:
      - `pm_a->force += f`
      - `pm_b->force -= f`

Then, integrate the new position for each point mass using Verlet integration. Algorithm:

2. For all `point_mass` in `point_masses`:
   1. Compute acceleration: `a = total_force / mass`
   2. Store the current position: `Vector3 temp = position`
   3. Update position using damped Verlet integration: 
      `position += (1 - damping) * (position - last_position) + a * delta_t^2`
   4. Update last position:`last_position = temp`

This approach approximates velocity as the difference between the current and previous positions. The `damping` term helps simulate energy loss due to internal friction, air resistance, or material properties.

### Result Gallery

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Default: Ks = 5000 N/m, density = 15 g/cm^2, Damping = 0.2
    </td>
  </tr>
  <tr>
    <td><img src="/images/ClothSim/image-20250407221227766.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407221245873.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407221315249.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Higher Ks = 50000 N/m
    </td>
  </tr>
  <tr>
    <td><img src="/images/ClothSim/image-20250408025916718.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250408025942221.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250408030037213.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Lower Ks = 500 N/m
    </td>
  </tr>
  <tr>
    <td><img src="/images/ClothSim/image-20250407221347387.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407221411910.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407221452428.png" style="width:100%;"></td>
  </tr>
</table>
> As the results show, a higher Ks (spring constant) results in a tighter cloth surface. Therefore, with a large Ks, the cloth appears flatter and more stretched, while with a smaller Ks, the surface becomes more wrinkled and uneven.


<table style="width:100%; table-layout: fixed;">
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Default: Ks = 5000 N/m, density = 15 g/cm^2, Damping = 0.2
    </td>
  </tr>
  <tr>
    <td><img src="/images/ClothSim/image-20250407221227766.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407221245873.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407221315249.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Higher Density = 150 g/cm^2
    </td>
  </tr>
  <tr>
    <td><img src="/images/ClothSim/image-20250407222448020.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407222523546.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407222600788.png" style="width:100%;"></td>
  </tr>
</table>
> When the mass of a point mass increases, it effectively weakens the restoring force of the springs. As a result, higher density leads to a more wrinkled cloth surface, while lower density results in a flatter appearance.

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Default: Ks = 5000 N/m, density = 15 g/cm^2, Damping = 0.2
    </td>
  </tr>
  <tr>
    <td><img src="/images/ClothSim/image-20250407221227766.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407221245873.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407221315249.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Higher Damping = 0.5
    </td>
  </tr>
  <tr>
    <td><img src="/images/ClothSim/image-20250407222830327.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407222851052.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407223006622.png" style="width:100%;"></td>
  </tr>
</table>

> A larger damping coefficient means that the cloth responds more slowly to external forces, resulting in a less noticeable increase in velocity compared to a smaller damping value. In the simulation, this manifests as the cloth falling more slowly.

<img src="/images/ClothSim/image-20250407224010844.png" alt="image-20250407224010844" style="zoom:50%;" />

> Converged pinned4.json, where four corners of the cloth are pinned.

## Part 3: Collisions with Other Objects

### Collision with Sphere

1. Check whether the point mass pm is colliding with the sphere:
   - Compute the distance between the point mass's current position (pm.position) and the sphere's origin;
   - If the distance is greater than or equal to the radius, no collision has occurred, so return immediately.

2. If a collision is detected, handle it as follows:
   1. Compute the tangent point on the sphere surface:
      - Normalize the vector (pm.position - origin), multiply by the radius;
      - Add the sphere origin to get the point on the sphere surface.
   2. Calculate the correction vector:
      - Subtract the previous position pm.last_position from the tangent point.
   3. Update the current position pm.position:
      - Set it to pm.last_position plus (1 - friction) times the correction vector;
      - This simulates friction, reducing the velocity after the collision and causing the point to slide along the sphere.

### Collision with Plane

1. Set up a ray from the previous position to the current position:
   - ray_origin = pm.last_position
   - ray_direction = unit vector from pm.last_position to pm.position
   - t_min = 0
   - t_max = length of the displacement vector

2. Check if the ray is parallel to the plane:
   - If dot(normal, ray_direction) is near zero, the ray is parallel;
   - In this case, there is no intersection, so return.

3. Compute the intersection point between the ray and the plane:
   - Use the plane equation and ray formula to solve for t_intersect:
     `t_intersect = dot((point - ray_origin), normal) / dot(normal, ray_direction)`
   - If t_intersect is outside [t_min, t_max], the segment doesn’t hit the plane, so return.

4. If a collision occurs:
   1. Calculate the intersection point with a slight offset to prevent sticking:
      - `intersection_point = ray_origin + ray_direction * (t_intersect - SURFACE_OFFSET)`
   
   2. Compute the correction vector:
      - `correction_vector = intersection_point - ray_origin`
   
   3. Update the point mass position:
         - `pm.position = pm.last_position + (1 - friction) * correction_vector`
              This simulates a collision response with friction, allowing the cloth to slide along the plane instead of bouncing or sticking unnaturally.

### Result Gallery

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td><img src="/images/ClothSim/image-20250407224454112.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407224525141.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407224557485.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td style="text-align:center;">Cloth Ks = 500</td>
    <td style="text-align:center;">Cloth Ks = 5000</td>
    <td style="text-align:center;">Cloth Ks = 50000</td>
  </tr>
</table>

> From the images, we observe that as the spring constant Ks increases, the cloth becomes more resilient and is better able to preserve its original shape, rather than completely draping and conforming to the surface of the sphere.

<img src="/images/ClothSim/image-20250407224900059.png" alt="image-20250407224900059" style="zoom:67%;" />

> A planar cloth lying restfully on a plane.

## Part 4: Self Collision

In each simulation step, we use spatial hashing to find point masses that are neighboring, and test if they collide.

### Hashing Spatial Points

1. Compute grid sizes for each axis:
   - `w` is a scaled width bin size based on the cloth resolution.
   - `h` is the height bin size.
   - `t` is the maximum of `w` and `h` to ensure cubic spacing.

2. Convert 3D position into discrete grid coordinates (x, y, z) by flooring the position values divided by the bin size.

3. Combine (x, y, z) using a set of large prime multipliers and XOR operations to generate a unique hash key.

### Building Spatial HashMap

1. Clear the existing spatial hash map:
   - Iterate through all entries in the map and delete their associated vector pointers.
   - Clear the map to remove all old entries.

2. Rebuild the spatial map for the current point mass positions:
   - For each point mass in the cloth:
     1. Compute its spatial hash using hash_position().
     2. If the hash key doesn't exist in the map, create a new vector for that bin.
     3. Add a pointer to the point mass into the appropriate spatial bin.

### Handling Self Collisions

1. Compute the hash key of the current point mass using its position.
   
2. If the hash bin is empty (no nearby particles), return early.

3. Initialize a correction vector and counter to accumulate collision responses.

4. For each candidate point mass in the same bin:
   - Skip if the candidate is the same as pm.
   - Compute the vector direction and distance between pm and the candidate.
   - If the distance is less than 2 × thickness (i.e., they are overlapping), compute the separation correction vector and accumulate it.
   - Increase the correction count.

5. If any collisions were found:
   - Average the correction vector across all overlaps.
   - Scale it by the number of simulation steps to distribute it smoothly.
   - Apply the correction to pm’s position.

### Result Gallery

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td><img src="/images/ClothSim/image-20250407225217549.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407225231703.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407225814208.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Falls and folds with default parameters (density = 15, Ks = 5000)
    </td>
  </tr>
  <tr>
    <td><img src="/images/ClothSim/image-20250407225947611.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407230003196.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407230030306.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Greater density (d = 150)
    </td>
  </tr>
  <tr>
    <td><img src="/images/ClothSim/image-20250407230123908.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407230131520.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250407230150792.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td colspan="3" style="text-align:center; font-weight:bold;">
      Greater spring coefficient (Ks = 50000)
    </td>
  </tr>
</table>

> The first set of images shows a piece of cloth falling onto itself and experiencing self-collision under normal parameters. In the second set, a higher point mass density is used, resulting in the cloth being compressed more significantly and appearing more tightly packed in its final state. The third set uses a larger spring constant, causing the cloth to better preserve its original shape and appear looser in the final state.

## Part 5: Shading

### Shaders Introduction

- A **shader program** is a small GPU-executed program used in the rendering pipeline. It consists of multiple shader stages (typically vertex and fragment shaders) and controls how geometry and pixels are processed on screen.

- A **vertex shader** operates on each vertex of a 3D object. It handles transformations (like model-view-projection), computes per-vertex data (such as normals or texture coordinates), and outputs information to the next stage of the pipeline.

- A **fragment shader** runs on each pixel (fragment) generated after rasterization. It determines the final color of a pixel, applying effects such as lighting, texturing, and shading.

### Blinn-Phong Shader

Blinn-Phong shading models consist of three components:

- **Ambient component** simulates indirect lighting. It adds a constant light to all surfaces, ensuring they are visible even without direct light.
- **Diffuse component** models light scattered evenly in all directions from a rough surface. It depends on the angle between the light direction and the surface normal.
- **Specular component** represents shiny highlights caused by direct reflection of light. In Blinn-Phong, it uses the half-vector between the light direction and view direction to calculate the intensity.

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td><img src="/images/ClothSim/image-20250408004540416.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250408004304470.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250408004338201.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250408004443470.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td style="text-align:center; font-weight:bold;">Full Blinn-Phong Model</td>
    <td style="text-align:center;">Specular</td>
    <td style="text-align:center;">Diffuse</td>
    <td style="text-align:center;">Ambient</td>
  </tr>
</table>
### Texture Shader

For texture models, we use:

+ A `uniform sampler2D in_texture` as the texture
+ `texture()` to do sampling in sampling space
+ Each pixel is shaded as 

The result is shown as:

<img src="/images/ClothSim/image-20250408010521298.png" alt="image-20250408010521298" style="zoom:67%;" />

> Texture (with Blinn-Phong shading to make it look fine)

### Bump & Displacement Shader

#### Bump Shader (Frag)

1. Construct the TBN matrix (Tangent, Bitangent, Normal):
   - `t`: Tangent vector from vertex input, normalized.
   - `b`: Bitangent is computed as the cross product of normal and tangent.
   - `n`: Normal vector, normalized.
   - `tbn`: 3x3 matrix that transforms bump-normal from tangent space to world/view space.
2. Compute bump-mapped normal:
   - Sample height values h(x, y) from a height function (e.g., texture or procedural).
   - Use finite differences in U and V directions to compute slope:
     - `dU` is the change in height along the horizontal axis.
     - `dV` is the change along the vertical axis.
     - Both are scaled by `u_height_scaling` and `u_normal_scaling`.
   - Construct a new tangent-space normal vector `n0 = (-dU, -dV, 1)`, and normalize it.
   - Transform `n0` using the TBN matrix to get the world/view space normal `nd`.
3. Compute diffuse/specular/ambient lighting (Blinn-Phong model):
   - Final output color is the sum of ambient, diffuse, and specular components.

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td><img src="/images/ClothSim/image-20250408011826325.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250408011805142.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center; font-weight:bold;">
      Bump Mapping (Default): normal = 2, height = 1
    </td>
  </tr>
</table>
#### Displacement Shader (Vertex)

1. Helper function:
   - `h(uv)`: Fetches the height value from the red channel of the height map texture at a given UV coordinate.

2. Main vertex processing:
   - Transform and normalize the vertex normal and tangent using the model matrix.
   - Pass UV coordinates unchanged.
   - Compute the displaced position:
     - Sample the height map at the UV coordinate.
     - Scale the normal vector by this height and the user-defined `u_height_scaling`.
     - Add this offset to the model-transformed vertex position to simulate bump displacement.
   - Output `gl_Position` by transforming the displaced position with the view-projection matrix.

<img src="/images/ClothSim/image-20250408011944897.png" alt="image-20250408011944897" style="zoom:67%;" />

> A rendered displacement + bumping shader, setting `normal=100`, `height=0.02`

#### Comparison

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td><img src="/images/ClothSim/image-20250408011805142.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250408011944897.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td style="text-align:center; font-weight:bold;">Bump Mapping</td>
    <td style="text-align:center; font-weight:bold;">Displacement Mapping</td>
  </tr>
</table>

> Comparison: Bump mapping doesn't disturb the geometry (vertices), while displacement mapping does. Basically, bump mapping is create an "illusion" of bump by disturbing normals used to render, while displacement mapping actually disturbs the vertex positions. Focus on the outline of the sphere and the differences are clear.

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td><img src="/images/ClothSim/image-20250408012112528.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250408012214986.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td style="text-align:center; font-weight:bold;">16 vertices lat/lng direction</td>
    <td style="text-align:center; font-weight:bold;">128 vertices lat/lng direction</td>
  </tr>
</table>

> With higher number of vertices on the sphere, the displacement mapping works better since it preserves more details of the height disturbance. However, for bump mapping, this doesn't matter since bump mapping don't care vertices.

### Environment Mapping

Explanation of mirror mapping fragment shader:

1. Compute viewing direction:
   - `view_dir = normalize(u_cam_pos - frag_pos)`
   - This is the direction from the fragment to the camera (eye).

2. Compute reflected direction:
   - `refl_dir = reflect(-view_dir, normal)`
   - The incoming view direction is reflected about the surface normal to simulate a mirror-like surface.

3. Sample from cube map:
   - `texture(u_texture_cubemap, refl_dir)` samples the reflected color from the environment.
   - This gives the illusion that the surface reflects its surroundings.

4. Output:
   - The final color (`out_color`) is simply the reflected environment color, making the surface look like a perfect mirror.

<table style="width:100%; table-layout: fixed;">
  <tr>
    <td><img src="/images/ClothSim/image-20250408012302065.png" style="width:100%;"></td>
    <td><img src="/images/ClothSim/image-20250408012309895.png" style="width:100%;"></td>
  </tr>
  <tr>
    <td style="text-align:center; font-weight:bold;">Cloth</td>
    <td style="text-align:center; font-weight:bold;">Sphere</td>
  </tr>
</table>
