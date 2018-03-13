#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --job-name=rgztest

module load tensorflow/1.4.0-py27-gpu
module load opencv openmpi

export PYTHONPATH=$PYTHONPATH:/home/wu082/software/lib/python2.7/site-packages

APP_ROOT=/home/wu082/proj/rgz-faster-rcnn

mpirun -np 1 python $APP_ROOT/tools/test_net.py \
                    --device gpu \
                    --device_id 0 \
                    --imdb rgz_2017_test22 \
                    --cfg $APP_ROOT/experiments/cfgs/faster_rcnn_end2end.yml \
                    --network VGGnet_test22 \
                    --weights $APP_ROOT/output/faster_rcnn_end2end/rgz_2017_train22/VGGnet_fast_rcnn-80000 \
                    --comp
