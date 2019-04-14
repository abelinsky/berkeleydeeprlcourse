#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    conda activate rl4
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ab/.mujoco/mjpro150/bin
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    cd Projects/rl/homework/hw1

    python run_expert.py experts/Humanoid-v2.pkl Ant-v2 --render --num_rollouts 100
    python run_expert.py experts/Ant-v2.pkl Ant-v2 --render --num_rollouts 50
    python run_expert.py experts/Walker2d-v2.pkl Walker2d-v2 --render --num_rollouts 200
    python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --render --num_rollouts 200

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle

import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import math

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()

            done = False
            totalr = 0.
            steps = 0

            print("\t === reset ===")

            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break

            print("\tReward = ", totalr)
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
