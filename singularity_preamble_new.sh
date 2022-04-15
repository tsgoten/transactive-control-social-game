#!/bin/bash
cd gym-microgrid/
(
flock -s 200
pip install -e .
) 200>lockfile
cd ..
cd gym-socialgame
(
flock -s 200
pip install -e .
) 200>lockfile

cd ..

#pip install tensorflow
#pip install keras
#pip install tensorflow-probability
#pip install tensorflow-gpu
