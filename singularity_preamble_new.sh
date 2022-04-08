#!/bin/bash
cd gym-microgrid/
pip install -e .
cd ..
cd gym-socialgame
pip install -e .
cd ..
pip install torch==1.0.0
#pip install tensorflow
#pip install keras
#pip install tensorflow-probability
#pip install tensorflow-gpu
