#!/bin/bash
lockfile packages.lock
cd gym-microgrid/
pip install -e .
cd ..
cd gym-socialgame
pip install -e .
cd ..
rm -f packages.lock
#pip install tensorflow
#pip install keras
#pip install tensorflow-probability
#pip install tensorflow-gpu
