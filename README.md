# 3D motion vectors and scene flow from Blender
This repository contains the code required to get forward or backward 3D motion vectors / scene flow from the Blender cycles renderer.


## How does it work?
While it is possible to extract forward and backward optical flow (2D motion vectors) from Blender using the [Vector pass](https://docs.blender.org/manual/en/latest/render/layers/passes.html), there is no direct way of exporting point-wise 3D motion vectors or scene flow.
For the project [Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo](https://spring-benchmark.org/), we developed a patch for Blender to export forward or backward **3D** motion vectors by reusing the vector pass.


## Setup
- Clone the Blender source code, e.g. via https://github.com/blender/blender. We tested the branch `blender-v2.80-release`.
- Apply the provided FW patch via `git apply <path to patch_v2.80-release_FW.patch>`
- Compile Blender; see https://wiki.blender.org/wiki/Building_Blender for details.

With the resulting blender executable, the export of forward 3D motion vectors (relative to the camera coordinate system) is possible.
The motion vectors are available in the Blender Vector pass (instead of optical flow).
In order to export backward 3D motion vectors, please use the patch `patch_v2.80-release_BW.patch`.

## Blender Scene Setup
- Make sure to select the cycles rendering engine and enable the vector pass in the "View Layer" menu. Note that the vector pass is only available if motion blur is turned off.
- Make sure that in the "Output" Menu under "Post Processing", "Compositing" is turned on, as well as "Use Nodes" in the "Compositing" view.
- In the "Composting" view add the following:
    - Add a "Separate RGBA" node and connect the "Render Layers"/Vector output to its input.
    - Add a "Combine RGBA" node and connect all 4 outputs of the "Separate RGBA" node to its corresponding inputs.
    - Add a "File Output" node and connect its input with the output of the "Combine RGBA" node.
    - In its node properties set the file format to "OpenEXR" and the color setting to RGBA Float (Full).

With this strategy, EXR files with 3D motion vectors in their R, G and B channels, respectively, are generated during rendering.
In order to also generate depth maps, report the above steps analogously for the Z-pass / Depth pass.

Additionally, extrinsic and intrinsic camera parameters are required, which can be extracted by running cameradata.py within the Blender scene.

## Conversion to scene flow
Given depth, 3D motion vectors as well as intrinsic and extrinsic camera parameters, scene flow and evaluation maps can be computed:
- Install the required python libraries numpy, OpenEXR and h5py. (Tested with Python 3.9)
- Use convert.py to generate scene flow data.


## Citation
If you make use of this code, please cite our paper:
```bibtex
@InProceedings{Mehl2023_Spring,
    author    = {Lukas Mehl and Jenny Schmalfuss and Azin Jahedi and Yaroslava Nalivayko and Andr\'es Bruhn},
    title     = {Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo},
    booktitle = {Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
```
