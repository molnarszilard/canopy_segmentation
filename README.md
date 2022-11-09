# Proximal Vine Canopy Segmentation based on Feature Pyramid Networks

In this repo we provide the code for our FPN-based canopy segmentation method.

## Preparation

You need to install pytorch (our version is 1.8.0), numpy, timeit, opencv, torchsummary. (conda environment is useful)

The dataset for training has to include the RGB images about the vines, and the corresponding masks, with the same name, but in different folder. These masks are stored as 3-channel black and white images (but if you have only 1-channel images, you can modify the dataloader.py file).

Inside the ```dataset``` folder, you need to have a folder called ```images```, and another called ```masks```. Inside these two folders distribute your dataset into a ```train``` and a ```test``` folder.

## Training

Check the possible arguments, then run:

```python train.py <args>```

To choose which GPU to run on run: ```CUDA_VISIBLE_DEVICES=0 python train.py <args>```

## Eval
In every case, you can specify a single file, or a folder (in which case every image in that folder will be analysed) to be processed (the program detects if the input is a folder or a file. However this is based on file extension, so if you don't use jpg, png or mp4, than specify your extensions in the scripts or convert your images). 

Match the arguments used in training for the evaluation, then to generate the masks, run:

```python eval.py <args>```

In video processing, you can set the sampling frame rate using the ```--frames``` argument.
If you have a video, and you would like to split that vieo into images, and mask them, run:

```python eval_video.py <args>```

If you want to create a new video from the masked images, run:

```python demo_video.py <args>```

## Demo

Full video are available at: https://youtu.be/IcPb2V716G8

### References

This code was mostly based on https://github.com/molnarszilard/ToFNest, while some aspects were taken from  https://github.com/MrD1360/deep_segmentation_vineyards_navigation