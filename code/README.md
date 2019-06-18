# Introduction
Train CNNs with weak bounding-box-level and image-level supervision for semantic segmentation.

__Panagiotis Meletis and Gijs Dubbelman (2019)__ _On Boosting Semantic Street Scene Segmentation with Weak Supervision._ The 30th IEEE Intelligent Vehicles Symposium (IV 2019), [full paper on arXiv](https://arxiv.org/abs/1903.03462).

Developed with Python 3.6, CUDA 9.2, Tensorflow GPU r1.12 + commit [b08c981](https://github.com/tensorflow/tensorflow/commit/b08c981d1d5556fb384cbbadcdc0d7c373324300), and [requirements.txt](requirements.txt).

# Prediction

Prerequisite: due to github space limitations download the checkpoints from [here](...) and unzip them into a directory (e.g. checkpoints).

To run inference on images:
```python
python predict.py log_dir problem_def predict_dir [optional arguments]
```

Required arguments:
* log_dir: the checkpoints directory, the system will use the latest checkpoint in this directory
* problem_def: one of the problem definition json files located in problem_definitions
* predict_dir: a directory containing images (RGB, formats: png, jpg, jpeg, ppm). The system will parse the directory for all supported images and will do sequential prediction on them.

Optional arguments:
* --plotting: live plotting
* --export_color_decisions: exports color label images at a specific directory (requires --results_dir to be provided)
* --results_dir \<dir>: provide this directory when exporting flags are provided

more arguments can be found at predict.py in function add_predict_arguments.

Example with Cityscapes checkpoint:
```python
python predict.py checkpoints/training80_step23033 problem_definitions/cityscapes/problem01.json sample_images --psp_module --plotting
```

# Base repository
Developed as an extension of the semantic segmentation system of this [repo](https://github.com/pmeletis/semantic-segmentation).

# Checkpoint performance
[Trained on Cityscapes and OpenScapes](...): Accuracy: 94.62, Mean Accuracy: 79.28, Mean IoU: 70.46.


###### Manual copy from internal hierarchical-semantic-segmentation-4/semantic-segmentation:v1.0-bboxes-image_labels-IV2019-github.