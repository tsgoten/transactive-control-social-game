#!/bin/bash
cd gym-microgrid/
(
flock -s 200
pip install -e .
) 200>lockfile
cd ..
cd gym-socialgame
pip install -e .
cd ..
(
flock -s 200
rm -f packages.lock
) 200>lockfile
#pip install tensorflow
#pip install keras
#pip install tensorflow-probability
#pip install tensorflow-gpu
