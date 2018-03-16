#!/bin/bash

# please change to your own python virtual environment path
source ~/pyml/bin/activate
RGZ_RCNN=/tmp/rgz_rcnn

python $RGZ_RCNN/tools/test_net.py \
                    --device cpu \
                    --device_id 0 \
                    --imdb rgz_2017_testD4 \
                    --cfg $RGZ_RCNN/experiments/cfgs/faster_rcnn_end2end.yml \
                    --network rgz_test \
                    --weights $RGZ_RCNN/data/pretrained_model/rgz/D4/VGGnet_fast_rcnn-80000 \
                    --comp
