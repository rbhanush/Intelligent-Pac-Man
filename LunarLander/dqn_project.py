from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import env_wrappers
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=True, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

'''
I have gotten mostly negative rewards(mainly due to aggressive landing) with a spattering of highly positive values 
and it never converged. 
Initially the bot seemed to be getting stuck at a local minimum (sort of) as it kept hovering 
without landing given the -100 for crashing. I assumed this would be due to fast decay of epsilon 
and experimented with many values of epsilon to finally arrive at this configuration, going from 500 learning steps
to even 100000 learning steps. Linearly decaying the value. 
'''

class DQN(object):
    """
    A starter class to implement the Deep Q Network algorithm

    TODOs specify the main areas where logic needs to be added.

    If you get an error a Box2D error using the pip version try installing from source:
    > git clone https://github.com/pybox2d/pybox2d
    > pip install -e .

    """

    def __init__(self, env):

        self.env = env
        self.sess = tf.Session()

        # A few starter hyperparameters
        self.batch_size = 512
        self.gamma = 0.9
        # If using e-greedy exploration
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 5000 # in learning steps
        # If using a target network
        self.clone_steps = 200
        self.eps_value_list = np.linspace(self.eps_start, self.eps_end, self.eps_decay)
        self.eps_value = self.eps_start
        # memory
        self.replay_memory = ReplayMemory(100000)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 10000

        self.cost_his = []
        self.eps_his = []
        self.reward_his = []
        self.state_space_size = self.env.observation_space.shape[0]
        self.action_space_size = self.env.action_space.n 
        self.lr = 0.001

        self.learn_step = 0
        
        # define yours training operations here...
        self.observation_input = tf.placeholder(tf.float32, shape=[None, self.state_space_size])
        self.observation_input_ = tf.placeholder(tf.float32, [None, self.state_space_size])
        self.build_model(self.observation_input)
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        
        # define your update operations here...

        self.num_episodes = 5000
        self.num_steps = 0




        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, observation_input, scope='train'):
        """
        TODO: Define the tensorflow model

        Hint: You will need to define and input placeholder and output Q-values

        Currently returns an op that gives all zeros.
        """
        self.q_target = tf.placeholder(tf.float32, [None, self.action_space_size])
        with tf.variable_scope('eval_net'):

            collection_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            l1_size = 10
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.state_space_size, l1_size], initializer=w_initializer, collections=collection_names)
                b1 = tf.get_variable('b1', [1, l1_size], initializer=b_initializer, collections=collection_names)
                l1 = tf.nn.relu(tf.matmul(self.observation_input, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [l1_size, self.action_space_size], initializer=w_initializer, collections=collection_names)
                b2 = tf.get_variable('b2', [1, self.action_space_size], initializer=b_initializer, collections=collection_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.losses.huber_loss(self.q_target, self.q_eval)
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        with tf.variable_scope('target_net'):
            collection_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.state_space_size, l1_size], initializer=w_initializer, collections=collection_names)
                b1 = tf.get_variable('b1', [1, l1_size], initializer=b_initializer, collections=collection_names)
                l1 = tf.nn.relu(tf.matmul(self.observation_input_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [l1_size, self.action_space_size], initializer=w_initializer, collections=collection_names)
                b2 = tf.get_variable('b2', [1, self.action_space_size], initializer=b_initializer, collections=collection_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def _reshape_state(self,state):
        return state.reshape(1,len(state))


    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement

        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.

        Currently returns a random action.
        """
        observation = obs[np.newaxis, :]
        if evaluation_mode:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.observation_input: observation})
            action = np.argmax(actions_value)
            return action
        if np.random.uniform() > self.eps_value:
            #print('Exploiting')
            actions_value = self.sess.run(self.q_eval, feed_dict={self.observation_input: observation})
            action = np.argmax(actions_value)
        else:
            #print('Exploring')
            action = self.env.action_space.sample()
        return action

    def update(self):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        """
        if self.learn_step % self.clone_steps == 0:
            self.sess.run(self.replace_target_op)

        batch_memory = self.replay_memory.sample(self.batch_size)
        observation_input_ = np.concatenate([[transition.next_state for transition in batch_memory]],axis=1)
        observation_input = np.concatenate([[transition.state for transition in batch_memory]],axis=1)


        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.observation_input_: observation_input_,
                self.observation_input: observation_input,
            })

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        batch_action = np.asarray([transition.action for transition in batch_memory])
        batch_reward = np.asarray([transition.reward for transition in batch_memory])
        eval_act_index = batch_action
        reward = batch_reward
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.observation_input: observation_input,
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        if self.learn_step < self.eps_decay:
            self.eps_value = self.eps_value_list[self.learn_step % self.eps_decay]
        self.eps_his.append(self.eps_value)
        self.learn_step += 1

    def plot_loss(self):
        import matplotlib.pyplot as plt
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(np.arange(len(self.cost_his)), self.cost_his)
        axarr[0].set_title('Learning Curve')
        axarr[1].plot(np.arange(len(self.cost_his)),self.eps_his)
        axarr[0].ylabel('Cost')
        axarr[1].ylabel('Epsilon')
        plt.xlabel('training steps')
        plt.show()

    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        done = False
        obs = env.reset()
        while not done:

            # self.eps_value = self.eps_value_list[eps_counter]
            action = self.select_action(obs,evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)
            self.replay_memory.push(obs,action,next_obs,reward,done)
            if (self.num_steps>self.min_replay_size) and (self.num_steps%50 == 0):
                self.update()
            obs = next_obs
            self.num_steps += 1

    def eval(self, save_snapshot=True):
        """
        Run an evaluation episode, this will call
        """
        total_reward = 0.0
        ep_steps = 0
        done = False
        obs = env.reset()
        print(self.eps_value)
        while not done:
            env.render()
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print ("Evaluation episode: ", total_reward)
        if save_snapshot:
            print ("Saving state with Saver")
            self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)

def train(dqn):
    dqn.num_steps = 0
    for i in xrange(dqn.num_episodes):
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            print(i)
            dqn.eval()
    dqn.plot_loss()

def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
