#!/bin/bash
cd gym-microgrid/ 
pip install -e . 
cd ..

cd gym-socialgame
pip install -e .
cd ..

pip install tensorflow 
pip install tensorflow-probability 
pip install tensorflow-gpu
pip install wandb
