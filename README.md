# Proximal Vine Canopy Segmentation based on Feature Pyramid Networks

In this repo, we provide the code for our FPN-based canopy segmentation method.

## Abstract

With the widening of the Agriculture 4.0 era, the use of autonomous robots in the agriculture field is becoming a priority. The key component of such an autonomous, often multi-robot system is the perception of the environment, which is based on 2D and 3D cameras. A base processing part of the 2D images is the segmentation of different zones in the images. This is the case also in the vineyards where in order to process complex plant canopies, segmenting the parts of the image containing the area of interest is a part of the pre-processing chain. In this work, we present a Feature Pyramid Network-based grape canopy segmentation method, which has great potential to create a segmentation mask, containing only the leaves and fruits of interest. We conducted our tests in different vineyards and we also obtained the above state-of-the-art segmentation results on public and custom datasets.

## Preparation

You need to install pytorch (our version is 1.8.0), numpy, timeit, opencv, torchsummary. (conda environment is useful)

The dataset for training has to include the RGB images of the vines, and the corresponding masks, with the same name, but in different folders. These masks are stored as 3-channel black and white images (but if you have only 1-channel images, you can modify the dataloader.py file).

Inside the ```dataset``` folder, you need to have a folder called ```images```, and another called ```masks```. Inside these two folders distribute your dataset into a ```train``` and a ```test``` folder.

## Training

Check the possible arguments, then run:

```python train.py <args>```

To choose which GPU to run on run: ```CUDA_VISIBLE_DEVICES=0 python train.py <args>```

## Eval
In every case, you can specify a single file, or a folder (in which case every image in that folder will be analyzed) to be processed (the program detects if the input is a folder or a file. However this is based on file extension, so if you don't use jpg, png, or mp4, then specify your extensions in the scripts or convert your images). 

Match the arguments used in training for the evaluation, then to generate the masks, run:

```python eval.py <args>```

In video processing, you can set the sampling frame rate using the ```--frames``` argument.
If you have a video, and you would like to split that video into images, and mask them, run:

```python eval_video.py <args>```

If you want to create a new video from the masked images, run:

```python demo_video.py <args>```

## Demo

Full video are available at: https://youtu.be/IcPb2V716G8

### References

This code was mostly based on https://github.com/molnarszilard/ToFNest, while some aspects were taken from  https://github.com/MrD1360/deep_segmentation_vineyards_navigation

## Citing

### BibTeX

```bibtex
@Article{molnar2023featurepyramidnetwork,
  author  = {Szilárd Molnár and Barna Keresztes and Levente Tamás},
  journal = {{IFAC-PapersOnLine}},
  title   = {{Feature Pyramid Network based Proximal Vine Canopy Segmentation}},
  year    = {2023},
  issn    = {2405-8963},
  note    = {22nd IFAC World Congress},
  number  = {2},
  pages   = {8920--8925},
  volume  = {56},
  doi     = {https://doi.org/10.1016/j.ifacol.2023.10.097},
  url     = {https://www.sciencedirect.com/science/article/pii/S240589632300441X},
}
```
