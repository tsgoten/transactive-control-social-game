#!/bin/bash
cd gym-microgrid/
setlock lockfile pip install -e .
cd ..
cd gym-socialgame
setlock lockfile pip install -e .
cd ..
#pip install tensorflow
#pip install keras
#pip install tensorflow-probability
#pip install tensorflow-gpu
