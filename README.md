# rgz_rcnn
Region-based Convolutional Neural Networks for Radio Galaxy Zoo.

This repository contains the source code of our proof-of-concept automated radio source morphology classifier based upon the Faster Region-based Convolutional Neutral Network ([Faster R-CNN](https://dl.acm.org/citation.cfm?id=3101780)). This is the first publicly available radio source morphology classifier that is capable of associating discrete and extended components of radio sources in an automated fashion. Hence, demonstrating the strengths for the application of deep learning algorithms in the automated analysis of radio sources that will eventuate from the next-generation large sky surveys at radio wavelengths.

Compared to existing Faster R-CNN implementations, we replaced the original RoI pooling layer with the [Spatial Transformer Network](https://arxiv.org/abs/1506.02025) (STN) pooling to support the end-to-end training. An *unexpected* benefit of this is that the code also runs on laptops that may not have GPUs (with a much longer latency  of course --- e.g. 6 seconds compared to 100s of milliseconds per image).

## Requirements

The code requires [Tensorflow 1.0 or above](https://www.tensorflow.org/install/), python 2.7 as well as the following python libraries:

* matplotlib
* numpy
* opencv-python
* scipy
* cython
* easydict
* astropy

Those modules can be installed using: `pip install -U pip` followed by `pip install -r requirements.txt`. It is **highly recommended** to setup a standalone [python virtual environment](https://virtualenv.pypa.io/en/stable/) to install these modules and run the code.

The code requires at least 700 MB of disk space and 3GB of RAM. For GPU training, this means the size of the device RAM is at least 3 GB.


## Setup

1. Clone this repository: `git clone https://github.com/chenwuperth/rgz_rcnn.git`
2. Compile: `cd lib` and `make`. This should compile a bunch of Cython/C code (for bounding box calculation), and will produce the dynamic libraries under both CPUs and GPUs (if available).
3. Download RGZ Data: `cd tools` and run `python download_data.py`. This will download data and RGZ model for training, testing, and demo.


## Tutorial

### Getting started

Run: `cd tools` and `python demo.py --radio ../data/rgzdemo/FIRSTJ011349.0+065025.fits --ir ../data/rgzdemo/FIRSTJ011349.0+065025_infrared.png` to detect a multi-component radio galaxy!

Run: `cd experiments/scripts` and `bash example_test_cpu.sh` to evaluate the RGZ model on 4603 images on your laptop using CPUs only. Please change the `RGZ_RCNN` environment variable in the script accordingly.

Run: `cd experiments/scripts` and `sbatch example_train_slurm.sh` to train your own RGZ model on GPU node managed by the SLURM job scheduler. You will need to change resources, paths, and module names based on the configuration of your own cluster.

## Questions?

Feel free to open an issue to discuss any questions not covered so far.

## Citation

If you benefit from this code, please cite our paper:

```
@article{wu2018rgzrcnn,
  title={Radio Galaxy Zoo: Region-based Convolutional Neural Networks for Source Detection and Morphology Classification},
  author={Wu, Chen and Wong, O. Ivy, and et al.},
  journal={in preparation},
  year={2018}
}
```
## Acknowledgement
The initial codebase was built on the awesome [Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF) project.
