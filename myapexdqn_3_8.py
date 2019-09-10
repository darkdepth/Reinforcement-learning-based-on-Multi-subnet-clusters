#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from atari_wrappers import wrap_deepmind
from time import sleep
import random
from replaymemory_1 import ReplayMemory
import operator
import sys
from segment_tree import SumSegmentTree, MinSegmentTree

GLOBAL_STEP = 0


# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def update_target_graph_part(from_scope, to_scope, iNdex=None):
    if iNdex == None:
        iNdex = random.randint(0, 1)
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope + '/' + str(iNdex))
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope + '/' + str(iNdex))

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def process_frame(image):
    image = np.reshape(image, [np.prod(image.shape)]) / 255.0
    return image


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class AC_Network():
    def __init__(self, sess, s_size, a_size, scope, trainer, _it_sum=None, _it_min=None):
        self.scope = scope
        self.i = 0
        self._it_sum, self._it_min = _it_sum, _it_min

        self.target_scope = 'worker_' + str(num_workers)
        with tf.variable_scope(self.scope):
            self.quantile = 1.0 / N
            self.cumulative_probabilities = (2.0 * np.arange(N) + 1) / (2.0 * N)
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.imageIn, num_outputs=32,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.conv1, num_outputs=64,
                                     kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.conv2, num_outputs=64,
                                     kernel_size=[3, 3], stride=[1, 1], padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv3), 512, activation_fn=tf.nn.relu)

            self.policy = slim.fully_connected(hidden, a_size * N,
                                               activation_fn=None,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(hidden, N,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)
            self.policyk = tf.reshape(self.policy, [-1, a_size, N])
            self.valuek = tf.reshape(self.value, [-1, 1, N])
            self.q = self.valuek + (self.policyk - tf.reduce_mean(self.policyk, axis=1, keep_dims=True))
            self.Q = tf.reduce_sum(self.q * self.quantile, axis=2)

            # if scope == 'worker_'+str(num_workers):
            if self.scope != self.target_scope:
                self.actions_q = tf.placeholder(shape=[None, a_size, N], dtype=tf.float32)
                self.q_target = tf.placeholder(shape=[None, N], dtype=tf.float32)
                self.ISWeights = tf.placeholder(shape=[None, N], dtype=tf.float32)

                self.q_actiona = tf.multiply(self.q, self.actions_q)
                self.q_action = tf.reduce_sum(self.q_actiona, axis=1)
                self.u = tf.abs(self.q_target - self.q_action)
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.u) * self.ISWeights, axis=1))
                tf.summary.scalar(self.scope + 'loss', tf.reduce_mean(self.loss))
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/')
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                # global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'worker_'+str(num_workers)+'/')

                part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.target_scope + '/')
                self.apply_grads_global = trainer.apply_gradients(zip(grads, part_vars))
                # self.apply_grads_global = trainer.apply_gradients(zip(grads,global_vars))
                # self.apply_grads_part = trainer.apply_gradients(zip(grads,part_vars))

    def train(self, sess, gamma):
        self.merged_summary = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, self.scope)
        )
        while not coord.should_stop():
            if len(replaymemory._storage) <= batch_size:
                continue
            episode_buffer, tree_idx, ISWeights = replaymemory.sample(batch_size, self._it_sum, self._it_min)
            episode_buffer = np.array(episode_buffer)
            observations = episode_buffer[:, 0]
            actions = episode_buffer[:, 1]
            rewards = episode_buffer[:, 2]
            observations_next = episode_buffer[:, 3]
            # print tree_idx
            Q_target = sess.run(self.Q, feed_dict={self.inputs: np.vstack(observations_next)})
            actions_ = np.argmax(Q_target, axis=1)
            action = np.zeros((batch_size, a_size))
            action_ = np.zeros((batch_size, a_size))
            for i in range(batch_size):
                action[i][actions[i]] = 1
                action_[i][actions_[i]] = 1
            action_now = np.zeros((batch_size, a_size, N))
            action_next = np.zeros((batch_size, a_size, N))
            for i in range(batch_size):
                for j in range(a_size):
                    for k in range(N):
                        action_now[i][j][k] = action[i][j]
                        action_next[i][j][k] = action_[i][j]
            q_target = sess.run(self.q_action, feed_dict={self.inputs: np.vstack(observations_next),
                                                          self.actions_q: action_next})
            q_target_batch = []
            for i in range(len(q_target)):
                qi = q_target[i]
                z_target_step = []
                for j in range(len(qi)):
                    z_target_step.append(gamma * qi[j] + rewards[i])
                q_target_batch.append(z_target_step)
            q_target_batch = np.array(q_target_batch)

            isweight = np.zeros((batch_size, N))
            for i in range(batch_size):
                for j in range(N):
                    isweight[i, j] = ISWeights[i]
            feed_dict = {self.q_target: q_target_batch,
                         self.inputs: np.vstack(observations),
                         self.actions_q: action_now,
                         self.ISWeights: isweight}

            result, l, abs_errors, _ = sess.run([self.merged_summary, self.loss, self.u, self.apply_grads_global],
                                                feed_dict=feed_dict)

            # print abs_errors
            writer.add_summary(result, self.i)
            self.i += 1
            abs_errors = np.mean(abs_errors, axis=1)

            # print(self._it_min)
            # print(self._it_sum)

            replaymemory.update_priorities(tree_idx, abs_errors, self._it_sum, self._it_min)

            # print(self._it_min)
            # print(self._it_sum)

            UPDATE_EVENT.clear()
            ROLLING_EVENT.set()


class Worker():
    def __init__(self, env, name, s_size, a_size, trainer, model_path, global_episodes, lock):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_mean_values = []
        self.lock = lock
        it_capacity = 1
        while it_capacity < max_memory:
            it_capacity *= 2
        self._it_sum = [SumSegmentTree(it_capacity)]
        self._it_min = [MinSegmentTree(it_capacity)]
        self.pre_t_m_loss = 1e5
        self.unpermit = True

        # self.replaymemory = ReplayMemory(max_memory)
        global worker_num
        # self.local_AC = AC_Network(sess, s_size, a_size, self.name, None)
        self.local_AC = AC_Network(sess, s_size, a_size, self.name, self.trainer, self._it_sum, self._it_min)
        worker_num += 1
        self.update_local_ops = update_target_graph(self.local_AC.target_scope, self.name)
        self.update_to_global_ops = update_target_graph(self.name, "worker_" + str(num_workers))

        self.update_ops = [[update_target_graph('worker_' + str(i), 'worker_' + str(j)) for j in range(num_workers + 1)]
                           for i in range(num_workers + 1)]
        self.update_part_ops = [
            [update_target_graph_part('worker_' + str(i), 'worker_' + str(j)) for j in range(num_workers + 1)] for i in
            range(num_workers + 1)]
        self.env = env

    def work(self, gamma, sess, coord, saver):
        global GLOBAL_STEP
        global rEward_dic
        global ALL_best
        global ALL_var
        global worker_num
        
        episode_count = sess.run(self.global_episodes)
        start_episode_count = episode_count
        total_steps = 0
        epsilon = 0.2
        the_rEward_dic = sorted(rEward_dic.iteritems(), key=operator.itemgetter(1), reverse=True)
        print ("Starting worker " + str(self.number))
        best_mean_episode_reward = -float('inf')
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                # sess.run(self.update_local_ops)
                if GLOBAL_STEP <= 500000:
                    sess.run(self.update_local_ops)
                elif GLOBAL_STEP % 100 == 0:
                    sess.run(self.update_local_ops)
                episode_reward = 0
                episode_step_count = 0
                d = False
                s = self.env.reset()
                s = process_frame(s)
                epsilon = epsilon * 0.995
                episode_step_count_for_loss = 0

                while not d:
                    if not ROLLING_EVENT.is_set():
                        ROLLING_EVENT.wait()
                    GLOBAL_STEP += 1

                    if random.random() > epsilon:
                        a_dist_list = sess.run(self.local_AC.Q, feed_dict={self.local_AC.inputs: [s]})
                        a_dist = a_dist_list[0]
                        a = np.argmax(a_dist)
                    else:
                        a = random.randint(0, a_size - 1)

                    s1, r, d, _ = self.env.step(a)

                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s

                    self.lock.acquire()
                    try:
                        replaymemory.add([s, a, r, s1, d], self._it_sum, self._it_min)
                    finally:
                        self.lock.release()
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    if GLOBAL_STEP > 500000 and GLOBAL_STEP % 1000 and worker_num == 33:
                        # range(num_workers/2 - 8,num_workers/2 + 8)
                        # beg = num_workers / 2 - 8
                        # end = num_workers / 2 + 8

                        beg = 0
                        end = num_workers
                        the_rEward_dic = sorted(rEward_dic.iteritems(), key=operator.itemgetter(1),
                                                reverse=True)

                        for i in range(3):
                            #do_again = True

                            Index_list_L = sorted(random.sample(range(int(beg), int(beg +  4)), 3 - i))
                            Index_list_R = sorted(random.sample(range(int(end - (i + 1) * 4), int(end - i * 4)), 3 - i))

                            for j in range(3 - i):
                                # L.acquire()
                                
                                tem_Index1 = int(the_rEward_dic[int(Index_list_L[j])][0].split('_')[1])
                                tem_Index2 = int(the_rEward_dic[int(Index_list_R[j])][0].split('_')[1])
                                
                                #do_again = True
                                sess.run(self.update_ops[tem_Index1][tem_Index2])
                                

                            # for j in range(3 - i):
                            #   tem_Index1[j] = int(the_rEward_dic[int(Index_list_L[j])][0].split('_')[1])
                            #   tem_Index2[j] = int(the_rEward_dic[int(Index_list_R[j])][0].split('_')[1])
                            #
                            #   locks_all[tem_Index2[j]][1] = 1
                            #   locks_all[tem_Index2[j]][0] = GLOBAL_STEP
                        """ 
                        for i in range(3):
                          for j in range(3 - i):
                                # L.acquire()
                                tem_Index1 = int(the_rEward_dic[int(Index_list_L[j])][0].split('_')[1])
                                tem_Index2 = int(the_rEward_dic[int(Index_list_R[j])][0].split('_')[1])
                                sess.run(self.update_ops[tem_Index1][tem_Index2])
                                #rEward_dic["worker_" + str(tem_Index2)] = rEward_dic["worker_" + str(tem_Index2)]
                                locks_all[tem_Index2][1] = 1
                                locks_all[tem_Index2][0] = GLOBAL_STEP
                                # L.release()
                        """
                        # rEward_dic[self.name] = np.mean(self.episode_rewards[-5:])
                        #print(33333333333333333333333333333333333333333333333333)
                        the_rEward_dic = sorted(rEward_dic.iteritems(), key=operator.itemgetter(1), reverse=True)
                        self.local_AC.target_scope = change_local_target(the_rEward_dic,self.name)
                    ##before is 1500
                    if GLOBAL_STEP > 500000 and GLOBAL_STEP % 1500 and worker_num == 33 and ALL_best > \
                            the_rEward_dic[0][1]:
                        # range(num_workers/2 - 8,num_workers/2 + 8)
                        # beg = num_workers / 2 - 8
                        # end = num_workers / 2 + 8

                        beg = 0
                        end = num_workers
                        
                        judge = True
                        the_rEward_dic = sorted(rEward_dic.iteritems(), key=operator.itemgetter(1),
                                                reverse=True)

                        for i in range(3):
                            # do_again = True

                            Index_list_L = sorted(random.sample(range(int(beg), int(beg + 4)), 3 - i))
                            Index_list_R = sorted(random.sample(range(int(end - (i + 1) * 4), int(end - i * 4)), 3 - i))

                            if judge:
                              for j in range(2):
                                # L.acquire()
                                
                                tem_Index1 = int(the_rEward_dic[int(Index_list_L[j])][0].split('_')[1])
                                
                                sess.run(self.update_ops[num_workers][tem_Index1])
                                
                              judge = False

                            for k in range(3 - i):
                                tem_Index2 = int(the_rEward_dic[int(Index_list_R[k])][0].split('_')[1])
                                
                                sess.run(self.update_ops[num_workers][tem_Index2])
                                # rEward_dic["worker_" + str(tem_Index2)] = ALL_best
                                
                        """
                        for i in range(3):
                            Index_list_L = sorted(random.sample(range(int(beg), int(beg + 4)), 3 - i))
                            Index_list_R = sorted(random.sample(range(int(end - (i + 1) * 4), int(end - i * 4)), 3 - i))
                            if judge:
                               for j in range(2):
                                  # L.acquire()
                                  tem_Index1 = int(the_rEward_dic[int(Index_list_L[j+1])][0].split('_')[1])
                                  sess.run(self.update_ops[num_workers][tem_Index1])
                                  #rEward_dic["worker_" + str(tem_Index1)] = ALL_best
                                  locks_all[tem_Index1][1] = 1
                                  locks_all[tem_Index1][0] = GLOBAL_STEP
                               judge = False   
                            for k in range(3 - i):
                                tem_Index2 = int(the_rEward_dic[int(Index_list_R[k])][0].split('_')[1])
                                sess.run(self.update_ops[num_workers][tem_Index2])
                                #rEward_dic["worker_" + str(tem_Index2)] = ALL_best
                                locks_all[tem_Index1][1] = 1
                                locks_all[tem_Index1][0] = GLOBAL_STEP
                        #print(222222222222222222222222222222222)
                        """
                        # rEward_dic[self.name] = np.mean(self.episode_rewards[-5:])
                        the_rEward_dic = sorted(rEward_dic.iteritems(), key=operator.itemgetter(1), reverse=True)
                        self.local_AC.target_scope = change_local_target(the_rEward_dic,self.name)

                    len_of_relay = len(replaymemory._storage)
                    if len_of_relay % batch_size == 0 and len_of_relay != 0 and d != True and worker_num == 33:
                        """
                        sess.run(self.update_ops[int(self.local_AC.target_scope.split('_')[1])][self.number])
                        if self.local_AC.target_scope == "worker_" + str(num_workers):
                            rEward_dic[the_rEward_dic[0][0]] = ALL_best
                        else:
                            rEward_dic[self.name] = rEward_dic[self.local_AC.target_scope]
                        """
                        #self.local_AC.target_scope = change_local_target(the_rEward_dic,self.local_AC.scope)
                        if GLOBAL_STEP <= 500000 or self.unpermit:
                            sess.run(self.update_local_ops)
                        elif GLOBAL_STEP//100 == 15:
                            # beg = num_workers / 2 - 8
                            # end = num_workers / 2 + 8
                            beg = 0
                            end = num_workers
                            for i in range(3):
                                Index_list_from = sorted(random.sample(range(int(beg), int(beg + 4)), 3 - i))
                                Index_list_to = sorted(
                                    random.sample(range(int(end - (i + 1) * 4), int(num_workers - i * 4)),
                                                  3 - i))

                                # Index_list_to = sorted(random.sample(range(i * 4, (i + 1) * 4), 4 - i))
                                # Index_list_from = sorted(
                                #    random.sample(range(num_workers - (i + 1) * 4, num_workers - i * 4), 4 - i))
                                for j in range(3 - i):
                                    # L.acquire()
                                    if random.random() * (num_workers + 1) <= 1:
                                        tem_Index1 = int(the_rEward_dic[int(Index_list_to[j])][0].split('_')[1])
                                        
                                        update_pg_op = \
                                            self.update_part_ops[num_workers][tem_Index1]
                                        # rEward_dic[the_rEward_dic[int(Index_list_to[j])][0]] = (ALL_best + rEward_dic[
                                        #     the_rEward_dic[int(
                                        #         Index_list_to[
                                        #             j])][
                                        #         0]]) / 2
                                    else:
                                        tem_Index1 = int(the_rEward_dic[int(Index_list_from[j])][0].split('_')[1])
                                        tem_Index2 = int(the_rEward_dic[int(Index_list_to[j])][0].split('_')[1])
                                        
                                        update_pg_op = \
                                            self.update_part_ops[tem_Index1][tem_Index2]
                                        # rEward_dic[the_rEward_dic[int(Index_list_to[j])][0]] = (rEward_dic[
                                        #                                                             the_rEward_dic[int(
                                        #                                                                 Index_list_from[
                                        #                                                                     j])][0]] +
                                        #                                                         rEward_dic[
                                        #                                                             the_rEward_dic[int(
                                        #                                                                 Index_list_to[
                                        #                                                                     j])][
                                        #                                                                 0]]) / 2

                                    sess.run(update_pg_op)

                                    self.unpermit = True
                                    #the_rEward_dic = sorted(rEward_dic.iteritems(), key=operator.itemgetter(1),
                                    #                        reverse=True)


                    


                    if d != True:
                        # sess.run(self.update_local_ops)
                        if GLOBAL_STEP <= 500000:
                            sess.run(self.update_local_ops)
                        elif GLOBAL_STEP % 100 == 0:
                            sess.run(self.update_local_ops)
                    else:
                        break

                if len(self.episode_rewards) > 5:
                    rEward_dic[self.name] = np.mean(self.episode_rewards[-5:])
                    the_rEward_dic = sorted(rEward_dic.iteritems(), key=operator.itemgetter(1), reverse=True)
                    if self.local_AC.i > 60000 and self.local_AC.i % 500:
                        #print(111111111111111111111111111111111111111)
                        print(self.local_AC.i)
                        self.local_AC.target_scope = change_local_target(the_rEward_dic,self.name)
                        

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)

                if episode_count % 15 == 0 and episode_count != 0 and episode_count - start_episode_count >= 15:
                    if self.name == 'worker_0' and episode_step_count_for_loss % 15 == 0:
                        now_t_m_loss = np.mean(self.episode_losses[-15:])
                        # print('\n episode: ', episode_count, 'global_step:', \
                        #      GLOBAL_STEP, 'mean_episode_loss: ', now_t_m_loss)
                        if self.pre_t_m_loss > now_t_m_loss:
                            self.unpermit = False
                        self.pre_t_m_loss = np.mean(self.episode_losses[-15:])
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if np.mean(self.episode_rewards[-7:]) >= ALL_best and np.var(
                        self.episode_rewards[-7:]) <= ALL_var and worker_num == 33:
                    ALL_best = np.mean(self.episode_rewards[-7:])
                    if ALL_best >= 20:
                        ALL_var = np.var(self.episode_rewards[-7:])
                    saver.save(sess, self.model_path + '/last-' + '_' + self.name + '.cptk')
                    print ("Saved Model")
                    print("All best reward is :" + str(ALL_best))
                    print("The name is :" + self.name)
                    sess.run(self.update_to_global_ops)
                if episode_count % 5 == 0 and episode_count != 0 and episode_count - start_episode_count >= 5:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', \
                              GLOBAL_STEP, 'mean_episode_reward: ', np.mean(self.episode_rewards[-5:]))
                        if self.episode_rewards == 'nan':
                            print('episode_rewards is ')
                            print(self.episode_rewards)

                    # if episode_count % 50 == 0 and self.name == 'worker_0':
                    #    saver.save(sess,self.model_path+'/last-'+str(episode_count)+'.cptk')
                    #    print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    if episode_count > 20 and best_mean_episode_reward < mean_reward:
                        best_mean_episode_reward = mean_reward

                episode_count += 1

    # def inject_summary(self, tag_dict):
    #     summary_str_lists = sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
    #         self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    #     })
    #     for summary_str in summary_str_lists:
    #         writer.add_summary(summary_str, GLOBAL_STEP)


def get_env(task):
    env_id = task.env_id
    # print(env_id)
    env = gym.make(env_id)
    env = wrap_deepmind(env)
    return env

def change_local_target(the_rEward_dic,name):
    t_idx = -1
    for t_id in range(len(the_rEward_dic)):
       if the_rEward_dic[t_id][0] == name:
           t_idx = t_id
    if t_idx == -1:
           
           print('something wrong')
           sys.exit(0)
                        
    return 'worker_' + str(the_rEward_dic[t_idx // 4][0].split('_')[1])

gamma = .99
s_size = 7056
load_model = False
model_path = './last'
N = 20
k = 1.

benchmark = gym.benchmark_spec('Atari40M')
# task = benchmark.tasks[3]
task = benchmark.tasks[1]
# task = benchmark.tasks[5]
tf.reset_default_graph()
ALL_best = 0
ALL_var = 2
if not os.path.exists(model_path):
    os.makedirs(model_path)

env = get_env(task)
a_size = env.action_space.n

global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=0.00015)
num_workers = 32

part_size = 4
batch_size = 10
max_memory = 50000
# max_memory = 5000
replaymemory = ReplayMemory(max_memory)
saver = tf.train.Saver(max_to_keep=num_workers)
lock = threading.Lock()
rEward_dic = {}

for i in range(num_workers):
    rEward_dic['worker_' + str(i)] = 0


with tf.Session() as sess:
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()

    GLOBAL_STEP = 0
    writer = tf.summary.FileWriter("logs/train", sess.graph)
    coord = tf.train.Coordinator()
    master_network = AC_Network(sess, s_size, a_size, 'worker_' + str(num_workers), trainer)  # Generate global network
    worker_num = 1
    workers = []
    for i in range(num_workers):
        env = get_env(task)
        workers.append(Worker(env, i, s_size, a_size, trainer, model_path, global_episodes, lock))

    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma, sess, coord, saver)
        t = threading.Thread(target=worker_work)
        t.start()
        # t.join()
        sleep(0.5)
        # sleep(1)
        worker_threads.append(t)
        # worker_threads.append(threading.Thread(target=master_network.train(sess, gamma)))
        """worker_threads.append(threading.Thread(target=worker.local_AC.train(sess, gamma)))"""
    count_i = 0
    single_cell = num_workers / part_size
    for worker in workers:
        if count_i % single_cell == 0:
            s = threading.Thread(target=worker.local_AC.train(sess, gamma))
            s.start()
            sleep(0.5)
            worker_threads.append(s)
        count_i += 1

        # for worker in workers:
    #     worker_threads.append(threading.Thread(target=worker.local_AC.train(sess, gamma)))

    # worker_threads[-1].start()
    coord.join(worker_threads)



