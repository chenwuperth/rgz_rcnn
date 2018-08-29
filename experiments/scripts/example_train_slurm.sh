#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --job-name=rgz_train

#module load tensorflow/1.4.0-py27-cpu

#module load tensorflow
#module load opencv openmpi

# if cuda driver is not in the system path, customise and add the following paths
# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#export PYTHONPATH=$PYTHONPATH:/home/astroimo/anaconda3/lib/python3.6
RGZ_RCNN=/mnt/c/Users/astroimo/icrar/rgz_rcnn

#mpirun -np 1 python $RGZ_RCNN/tools/train_net.py \
python $RGZ_RCNN/tools/train_net.py \
                    --device cpu \
                    --device_id 0 \
#                    --imdb rgz_2017_trainD4 \
#                    --iters 80000 \
                    --imdb rgz_2017_testD4 \
                    --iters 10 \
                    --cfg $RGZ_RCNN/experiments/cfgs/faster_rcnn_end2end.yml \
                    --network rgz_train \
                    --weights $RGZ_RCNN/data/pretrained_model/imagenet/VGG_imagenet.npy
