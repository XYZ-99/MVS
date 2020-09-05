#!/usr/bin/env bash
conda create -n Photo2Model python=3.6
source activate Photo2Model
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch
conda install -c anaconda pillow
pip install -r requirements
