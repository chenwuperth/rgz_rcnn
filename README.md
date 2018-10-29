# ClaRAN
[Radio Galaxy Zoo: ClaRAN - A Deep Learning Classifier for Radio Morphologies](https://arxiv.org/abs/1805.12008)

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

Those modules can be installed using: `pip install -U pip` followed by `pip install -r requirements.txt`. It is **highly recommended** to setup a standalone [python virtual environment](https://pypi.python.org/pypi/virtualenv) to install these modules and run the code.

The code requires at least 3 GB of disk space (to store images and pre-trained models) and 3GB of RAM during training. For GPU training, this means the size of the device RAM is at least 3 GB.


## Setup

1. Clone this repository: `git clone https://github.com/chenwuperth/rgz_rcnn.git`
2. Compile: `cd lib` and `make`. This should compile a bunch of Cython/C code (for bounding box calculation), and will produce the dynamic libraries under both CPUs and GPUs (if available).
3. Download RGZ Data: `cd tools` and run `python download_data.py`. This will download data and RGZ model for training, testing, and demo.


## Tutorial

### Getting started

**Run**: `cd tools` and `python demo.py --radio ../data/rgzdemo/FIRSTJ011349.0+065025.fits --ir ../data/rgzdemo/FIRSTJ011349.0+065025_infrared.png` to detect a multi-component radio galaxy! Some examples of demo output are shown below:

<img src="http://ict.icrar.org/store/staff/cwu/rgz_data/demo_result.png" width="800">

Each detected box denotes an identified radio source, and its morphology is succinctly labelled as *X* C_*Y* P, where *X* and *Y* denotes the number of radio components and the number of radio peaks respectively. Each morphology label is also associated with a score between 0 and 1, indicating the probability of a morphology class being present.

**Run**: `cd experiments/scripts` and `bash example_test_cpu.sh` to evaluate the RGZ model on 4603 images on your laptop using CPUs only. Please change the `RGZ_RCNN` environment variable in the script accordingly. The output records the [Average Precision](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision) for each class and the overall [mean AP](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) for the *D4* method.

| Morphology Classes       | AP     |
|-------------|--------|
| 1C_1P       | 0.8784 |
| 1C_2P       | 0.7074 |
| 1C_3P       | 0.8941 |
| 2C_2P       | 0.8200 |
| 2C_3P       | 0.7916 |
| 3C_3P       | 0.9269 |
| mAP         | 83.6% |

**Run**: `cd experiments/scripts` and `sbatch example_train_slurm.sh` to train your own RGZ model on GPU node managed by the SLURM job scheduler. You will need to change resources, paths, and module names based on the configuration of your own cluster.

## Questions?

Feel free to open an issue to discuss any questions not covered so far.

## Citation

If you benefit from this code, please cite our paper:

```
@article{wu2018rgzrcnn,
  title={Radio Galaxy Zoo: ClaRAN — a deep learning classifier for radio morphologies},
  author={Wu C, Wong O. i., Rudnick L., Shabala S. S., Alger M. J., Banfield, J. K., Ong C. S., White S. V., Garon A. F., Norris R. P., Andernach H. A., Tate, J., Lukic V., Tang H., Schawinski K., Diakogiannis F. I.},
  journal={Monthly Notices of the Royal Astronomical Society},
  month={October},
  year={2018}
}
```
## Acknowledgement
The initial codebase was built on the awesome [Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF) project.
