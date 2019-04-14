#!/bin/bash

conda activate rl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ab/.mujoco/mjpro150/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
cd Projects/rl/homework/hw1

python imitationlearning.py experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 20