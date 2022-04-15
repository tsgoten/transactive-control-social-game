#!/bin/bash
cd gym-microgrid/
(
flock -e ../lockfile
pip install -e .
)
cd ..
cd gym-socialgame
(
flock -e ../lockfile
pip install -e .
)

cd ..

#pip install tensorflow
#pip install keras
#pip install tensorflow-probability
#pip install tensorflow-gpu
