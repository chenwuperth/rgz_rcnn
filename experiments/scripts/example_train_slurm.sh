#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --job-name=rgz_train

module load tensorflow/1.4.0-py27-gpu
module load opencv openmpi

export PYTHONPATH=$PYTHONPATH:/home/wu082/software/lib/python2.7/site-packages
RGZ_RCNN=/flush1/wu082/proj/rgz_rcnn
mpirun -np 1 python $RGZ_RCNN/tools/train_net.py \
                    --device gpu \
                    --device_id 0 \
                    --imdb rgz_2017_trainD4 \
                    --iters 80000 \
                    --cfg $RGZ_RCNN/experiments/cfgs/faster_rcnn_end2end.yml \
                    --network rgz_train \
                    --weights $RGZ_RCNN/data/pretrained_model/imagenet/VGG_imagenet.npy
