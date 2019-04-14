import os
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import time
import shutil
import tf_util
import gym
import load_policy

from TFModel import TFModel
from KerasModel import KerasModel


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

    # loading expert policy
    policy_fn = load_policy.load_policy(args.expert_policy_file)

    def tf_model(layers_dims):
        return TFModel(layers_dims=layers_dims)

    def keras_model(layers_dims):
        return KerasModel(layers_dims=layers_dims, batch_size=1024)

    # training behaviuor cloning policy
    def train_policy(env, neural_net_layers_dims, model_fn=tf_model, num_epochs=50):
        """Trains policy's neural net.

        Args:
          neural_net_layers_dims: Neural net structure. For example, [128, 128, 64]

        Returns:
          Trained model.
        """
        print("Training policy")

        # load data
        def load_data(filepath, stack_to=None):
            with open(filepath, "rb") as f:
                expert_data = pickle.loads(f.read())

            assert expert_data is not None, "Demontrations were not founded"
            observations = expert_data['observations']
            actions = expert_data['actions'][:, 0]

            assert observations.shape[0] == actions.shape[0], "Shapes do not match"
            return observations.astype(np.float32), actions.astype(np.float32)

        observations, actions = load_data('expert_data/{}.pkl'.format(env))
        x_train, x_test, y_train, y_test = train_test_split(observations, actions, test_size=0.1)
        n_inputs, n_outputs = x_train.shape[1], y_train.shape[1]
        layers_dims = np.concatenate(([n_inputs], neural_net_layers_dims, [n_outputs]))
        # model = TFModel(layers_dims=layers_dims)
        model = model_fn(layers_dims=layers_dims)
        model.build_graph((x_train, y_train), (x_test, y_test))
        start_time = time.time()
        model.train(num_epochs)
        print('Training time: {0} seconds'.format(time.time()-start_time))
        return model

    # start interacting with environment
    env = args.envname  # for example, "Humanoid-v2"
    bc_policy = train_policy(env, [64, 64, 64], model_fn=tf_model, num_epochs=100)

    num_dagger_steps = 0

    with tf.Session() as sess:
        tf_util.initialize()

        bc_policy.restore_model(sess)

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        for dagger_step in range(1+num_dagger_steps):
            returns = []
            observations = []
            dagger_actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()

                done = False
                totalr = 0.
                steps = 0

                while not done:
                    observation = obs[None, :]
                    obs = np.array(observation).astype(np.float32)
                    action = bc_policy.predict(obs, sess).transpose()

                    expert_action = policy_fn(observation)
                    observations.append(obs)
                    dagger_actions.append(expert_action)

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

            # perform DAGGer-training
            if dagger_step < num_dagger_steps-1:
                print(' === Performing DAGGer step #{0} == '.format(dagger_step))
                bc_policy.add_data(train_data=(np.array(observations).squeeze(), np.array(dagger_actions).squeeze()))
                bc_policy.train(n_epochs=50)


if __name__ == '__main__':
    main()
