
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

___

## What is KαLOS

**KαLOS** is a professional toolkit for evaluating dataset quality in Computer Vision tasks (BBox, Segm, Keypoints, 3D).
It is designed to be modular, provide analytical depth, and be easily extensible to new tasks.

To use KαLOS, two requirements need to be met:
1) Tasks should combine localization and classification to be eligible for extension.
2) Datasets should contain at least two raters for the scene/image/data(-subset) you want to evaluate. 
Raters do not have to be human. Data can also be obtained via semi-automated labeling or auto-labeling.

<p align="center">
  <img src="docs/assets/kalos_processing_v4.svg" width="100%" />
</p>

___

## Installation
Simply run:
```
pip install kalos
```

<details>
<summary>Editable install instructions (if you want to add another task)</summary>
In case you have a task not yet covered by KαLOS, you will need to add a distance/similarity function 
in `src/kalos/similarity_functions.py`. You are welcome to make a merge request to add this function
to the repository after thorough testing.

Clone the repository:
```
https://github.com/Madave94/kalos.git
```
Move into the folder and make an editable install:
```
    cd kalos
```
Use your build system of choice. Pip:
```
    pip install -e .
```
UV (Recommended):
```
    uv sync
```
</details>

___

## Quickstart
KαLOS evaluation follows four steps. A general example looks like:
1) Calculate expected and observed disagreement:
```
    kalos calc-disagreement --config path/to/config/file.yaml
```
2) Run principled configuration to identify the optimal localization threshold ($\tau^*$):
```
    kalos configure --config path/to/config/file.yaml
```
3) Run agreement calculation:
```
    kalos execute --config path/to/config/file.yaml
```
4) (Optional) Plot diagnostics:
```
    kalos plot --config path/to/config/file.yaml
```

**Note on Portability:** All paths in YAML configs are relative to the config file itself (this comes from the jsonargparse design). 
You can move config folders together with datasets and results without breaking the pipeline.

<details>
<summary>Full Walkthrough: Hello World with TexBiG</summary>
This example uses the TexBiG dataset to demonstrate how to derive the distance/similarity function and the configuration anchor,
as well as run the main calculation and downstream analysis. These results are also shown in the CVPR paper Section 14.3.

1. Download the formatted annotations [TexBiG](https://drive.google.com/file/d/1RKJ7EhmRRCKySp_kKpmMCLN_v9TrElRy/view?usp=sharing).
2. Move them into the folder `datasets`, so that the relative paths work.
3. Calculate expected and observed disagreements. To showcase how different distance/similarity functions are compared,
calculating this for three different functions should be done:
```
    kalos calc-disagreement --config configs/instance_segmentation/texbig/texbig_segm_centroid_disagreement.yaml
    kalos calc-disagreement --config configs/instance_segmentation/texbig/texbig_segm_iou_disagreement.yaml
    kalos calc-disagreement --config configs/instance_segmentation/texbig/texbig_segm_giou_disagreement.yaml
```
4. Run principled configuration:
```
    kalos configure --config configs/instance_segmentation/texbig/texbig_segm_configure.yaml
```
The principled configuration provides a calibration anchor for each distance/similarity function and a KS statistic which
provides information as to which of these functions is the best separator.
**Note**: To stay consistent with existing work, the disagreement evaluation requires a distance function, while the remaining
pipeline uses a similarity function. The functions are mostly the same as $d=1-s$ and vice-versa. Keep this
in mind when you select the threshold values. Logging will explicitly hint you towards this.
5. Run KαLOS execution:
```
    kalos execute --config configs/instance_segmentation/texbig/texbig_segm_kalos.yaml
```
After the run finishes, the results are stored in `results/instance_segmentation/texbig`.
6. Plot the results:
```
    kalos plot --config configs/instance_segmentation/texbig/texbig_segm_kalos.yaml
```
This creates the plot used in chapter 14.1 in the CVPR paper.

</details>

<details>
<summary>Run using CLI instead</summary>
Instead of `kalos`, you can also call `/src/kalos/cli.py`. The same entrypoint is used.
</details>


<details>
<summary>API usage example</summary>
If you want to include KαLOS into an existing library, the core function to call is `calculate_iaa` in `src/kalos/core.py`.
You will likely need to combine this with the preprocessing from `preprocess_data` in `src/kalos/correspondence/correspondence_algorithms.py`,
which provides you information about the data structure you need to input for KαLOS.
</details>

___

## Prepare your own dataset

Currently KαLOS supports the following tasks with the following similarity or distance functions:

| Task                                | Similarity/Distance functions |
|-------------------------------------|-------------------------------|
| Object Detection                    | IoU, GIoU, Centroid-Distance  |
| Instance Segmentation               | IoU, GIoU, Centroid-Distance  |
| 3D Volumetric Instance Segmentation | 3D-IoU                        |
| Keypoints / Pose Estimation         | IN-MPJPE                      |

Besides the regular information in your annotation, data should contain two additional pieces of information:
1) `rater_id` inside each annotation, specifying the responsible rater.
2) `rater_list` inside the image/scene, specifying the raters assigned to a specific image/scene.

<details>
<summary>Show format example</summary>
For the COCO-JSON format currently present in the code, data might look like:

Annotations:
```
 {
     "id": 1,
     "image_id": 1,
     "category_id": 3,
     "bbox": [x, y, width, height],
     "area": area,
     "iscrowd": 0,
     "rater_id": "r1"
 }
```

Images:
```
 {
     "id": 1,
     "file_name": "image1.jpg",
     "height": height,
     "width": width,
     "rater_list": ["r1", "r2"]
 }
```
</details>

<details>
<summary>Extending to new tasks</summary>
This is advanced functionality. It requires adding a new similarity function. You should install the repository in
editable mode and look into the docstrings. Feel free to create an issue if you are stuck, and if you finish
the task expansion, you are welcome to make a merge request.

KaLOS is built on a Hexagonal architecture, making it easy to add new geometric tasks (e.g., 3D Point Clouds, Audio Segments):
1. **Similarity Function:** Register your new metric in `src/kalos/iaa/similarity_functions.py` (e.g., `my_custom_iou`).
2. **Configuration:** Add your task name to the `Literal['bbox', ...]` types in `src/kalos/config.py`.
3. **Registry:** Add your metric to the `SIMILARITY_FUNCTIONS` registry.

The core math engine will automatically pick up your new metric for all agreement tiers.

</details>

___

## Citation

```bibtex
@inproceedings{tschirschwitz2026kalos,
  title={KαLOS finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks},
  shorttitle = {KαLOS},
  author = {Tschirschwitz, David and Rodehorst, Volker},
  booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year = {2026}
}
```
