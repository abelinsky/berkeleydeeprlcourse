# !/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ab/.mujoco/mjpro150/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf -e 1
# python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reparam -e 1

# -- Humanoid-v2
# python train_mujoco.py --env_name Humanoid-v2 --exp_name reparam -e 1

# python train_mujoco.py --env_name Ant-v2 --exp_name reparam -e 3
# python train_mujoco.py --env_name Ant-v2 --exp_name reparam_2qf -e 3

# python plot.py data/sac_HalfCheetah-v2_reinf_25-05-2019_15-26-59/1 --value LastEpReturn

