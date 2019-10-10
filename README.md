# ClaRAN
[Radio Galaxy Zoo: ClaRAN - A Deep Learning Classifier for Radio Morphologies](https://academic.oup.com/mnras/article/482/1/1211/5142869)

**ClaRAN** - **Cla**ssifying **R**adio sources **A**utomatically with **N**eural networks - is a proof-of-concept radio source morphology classifier based upon the Faster Region-based Convolutional Neutral Network ([Faster R-CNN](https://dl.acm.org/citation.cfm?id=3101780)). *ClaRAN* is the first publicly available radio source morphology classifier that is capable of associating discrete and extended components of radio sources in an automated fashion. *ClaRAN* demonstrates the feasibility of applying deep learning methods for cross-matching complex radio sources of multiple components with infrared maps. The promising results from *ClaRAN* have implications for the further development of efficient cross-wavelength source identification, matching, and morphology classifications for future radio surveys.

*ClaRAN* replaces the original RoI cropping layer with the [Spatial Transformer Network](https://arxiv.org/abs/1506.02025) (STN) pooling to support a fully end-to-end training pipeline. An *unexpected* benefit of this is that the code also runs on laptops that may not have GPUs (with a much longer latency  of course --- e.g. 6 seconds compared to 100s of milliseconds per image).

## Requirements

The code requires [Tensorflow 1.0 or above](https://www.tensorflow.org/install/), python 2.7 as well as the following python libraries:

* matplotlib
* numpy
* opencv-python
* scipy
* cython
* easydict
* astropy
* Pillow
* pyyaml

**Install Modules:** 

`pip install -U pip`

`pip install -r requirements.txt`

It is **highly recommended** to setup a standalone [python virtual environment](https://pypi.python.org/pypi/virtualenv) to install these modules and run the code.

The code requires at least 3 GB of disk space (to store images and pre-trained models) and 3GB of RAM during training.


## Setup

* Clone this repository: 

  `git clone https://github.com/chenwuperth/rgz_rcnn.git`

* Compile: 

  `cd lib`

  `make`

This should compile a bunch of [Cython](https://cython.org/)/ C code (for bounding box calculation), and will produce the dynamic libraries under both CPUs and GPUs (if available).

* Download RGZ Data: 

  `cd tools` 

  `python download_data.py` 

This will download data and RGZ model for training, testing, and demo.


## Tutorial

### Getting started

#### To detect a multi-component radio galaxy!:
**Run:**

`cd tools`

`python demo.py --radio ../data/rgzdemo/FIRSTJ011349.0+065025.fits --ir ../data/rgzdemo/FIRSTJ011349.0+065025_infrared.png` 

Some examples of demo output are shown below:

<img src="http://ict.icrar.org/store/staff/cwu/rgz_data/demo_result.png" width="800">

Each detected box denotes an identified radio source, and its morphology is succinctly labelled as *X* C_*Y* P, where *X* and *Y* denotes the number of radio components and the number of radio peaks respectively. Each morphology label is also associated with a score between 0 and 1, indicating the probability of a morphology class being present.

#### To evaluate the RGZ model on 4603 images on your laptop using CPUs only:
**Run:**

`cd experiments/scripts` 

`bash example_test_cpu.sh`  

Please change the `RGZ_RCNN` environment variable in the script accordingly. The output records the [Average Precision](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision) for each class and the overall [mean AP](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) for the *D4* method.

| Morphology Classes       | AP     |
|-------------|--------|
| 1C_1P       | 0.8784 |
| 1C_2P       | 0.7074 |
| 1C_3P       | 0.8941 |
| 2C_2P       | 0.8200 |
| 2C_3P       | 0.7916 |
| 3C_3P       | 0.9269 |
| mAP         | 83.6% |

#### To train your own RGZ model on GPU node managed by the SLURM job scheduler:
**Run:**

`cd experiments/scripts`

`sbatch example_train_slurm.sh` 

You will need to change resources, paths, and module names based on the configuration of your own cluster.

## Questions?

Feel free to open an issue to discuss any questions not covered so far.

## Citation

If you benefit from this code, please cite our [paper](https://academic.oup.com/mnras/article/482/1/1211/5142869):

```
@article{doi:10.1093/mnras/sty2646,
author = {Wu, Chen and Wong, Oiwei Ivy and Rudnick, Lawrence and Shabala, Stanislav S and Alger, Matthew J and Banfield, Julie K and Ong, Cheng Soon and White, Sarah V and Garon, Avery F and Norris, Ray P and Andernach, Heinz and Tate, Jean and Lukic, Vesna and Tang, Hongming and Schawinski, Kevin and Diakogiannis, Foivos I},
title = {Radio Galaxy Zoo: ClaRAN – a deep learning classifier for radio morphologies},
journal = {Monthly Notices of the Royal Astronomical Society},
volume = {482},
number = {1},
pages = {1211-1230},
year = {2019},
doi = {10.1093/mnras/sty2646},
URL = {http://dx.doi.org/10.1093/mnras/sty2646}
}
```
## Acknowledgement
The initial codebase was built on the awesome [Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF) project.
