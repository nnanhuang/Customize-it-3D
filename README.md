# Customize-It-3D: High-Quality 3D Creation from A Single Image Using Subject-Specific Knowledge Prior

<!-- ![Teaser](teaser.png) -->
<p align="center">
  <img src="./assets/teaser.jpg" width=800>
</p>

[Nan Huang](https://github.com/nnanhuang),
[Ting Zhang](https://www.microsoft.com/en-us/research/people/tinzhan/),
[Yuhui Yuan](https://www.microsoft.com/en-us/research/people/yuyua/),
[Dong Chen](https://www.microsoft.com/en-us/research/people/doch/),
[Shanghang Zhang](https://www.shanghangzhang.com/)

[![arXiv](https://img.shields.io/badge/ArXiv-2310.11784-red)](http://arxiv.org/abs/2312.11535)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://nnanhuang.github.io/projects/customize-it-3d/)

<p align="center">
  <img src="./assets/pipeline.jpg" width=800>
</p>

<div align="center">
<b>Pipeline.</b> We propose a two-stage framework Customize-It-3D for high-quality 3D creation from a reference image with subject-specific diffusion prior. We first cultivate subject-specific knowledge prior using multi-modal information to effectively constrain the coherency of 3D object with respect to a particular identity. At the coarse stage, we optimize a NeRF for reconstructing the geometry of the reference image in a shading-aware manner. We further build point clouds with enhanced texture from the coarse stage, and jointly optimize the texture of invisible points and a learnable deferred renderer to generate realistic and view-consistent textures.
</div>

## News
- [2023/12/14] Our code will open soon. 

## Demo of 360° geometry
<img src="assets/resfull.gif" width="800" />

## Bibtex
If you find this work useful, a citation will be appreciated via:

<pre><code>
    @misc{huang2023customizeit3d,
      title={Customize-It-3D: High-Quality 3D Creation from A Single Image Using Subject-Specific Knowledge Prior}, 
      author={Nan Huang and Ting Zhang and Yuhui Yuan and Dong Chen and Shanghang Zhang},
      year={2023},
      eprint={2312.11535},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
</code></pre>

## Acknowledgments
This code borrows heavily from [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion), many thanks to the author. 
