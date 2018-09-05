#!/bin/bash

#SBATCH --partition=mlgpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=rgz_train


# if cuda driver is not in the system path, customise and add the following paths
# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PYTHONPATH=$PYTHONPATH:/home/chen/software/lib/python2.7/site-packages
RGZ_RCNN=/home/yuno/intern/rgz_rcnn
/usr/bin/python python $RGZ_RCNN/tools/train_net.py \
                    --device gpu \
                    --device_id 0 \
                    --imdb rgz_2017_allD1 \
                    --iters 80000 \
                    --cfg $RGZ_RCNN/experiments/cfgs/faster_rcnn_end2end.yml \
                    --network rgz_train \
                    --weights $RGZ_RCNN/data/pretrained_model/imagenet/VGG_imagenet.npy
