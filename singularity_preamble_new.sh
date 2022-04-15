#!/bin/bash
cd gym-microgrid/
(
flock -x 200
rm -rf ~/.local && pip install -e .
) 200> ../lockfile
cd ..
cd gym-socialgame
(
flock -x 200
rm -rf ~/.local && pip install -e .
) 200> ../lockfile

cd ..

#pip install tensorflow
#pip install keras
#pip install tensorflow-probability
#pip install tensorflow-gpu
