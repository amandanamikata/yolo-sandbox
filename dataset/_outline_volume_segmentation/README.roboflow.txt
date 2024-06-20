
outline_volume_segmentation - v2 2024-06-18 6:42pm
==============================

This dataset was exported via roboflow.com on June 19, 2024 at 12:21 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 805 images.
123-ba8C-XdT9-MQ36 are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random Gaussian blur of between 0 and 2.5 pixels

The following transformations were applied to the bounding boxes of each image:
* Random Gaussian blur of between 0 and 2.5 pixels


