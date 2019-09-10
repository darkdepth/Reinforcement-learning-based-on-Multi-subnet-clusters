#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ily 
"""
import numpy as np
import random

class ReplayMemory:
    def __init__(self, replay_size, alpha=0.6):
        self.replay_size = replay_size
        self.cnt = 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < replay_size:
            it_capacity *= 2

        #self._it_sum = SumSegmentTree(it_capacity)
        #self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._storage = []
        self._maxsize = replay_size
        self._next_idx = 0

    def add(self, data,_it_sum,_it_min):
        #new_data = []
        #for i in data:
        #    i.wait_to_read()
        #    new_data.append(copyto(i))
        
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            #print self._storage
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        idx = self._next_idx
        _it_sum[0][idx] = self._max_priority ** self._alpha
        _it_min[0][idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size,_it_sum):
        res = []
        for _ in range(batch_size):
            mass = random.random() * _it_sum[0].sum(0, len(self._storage) - 1)
            idx = _it_sum[0].find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size,_it_sum,_it_min, beta=0.4):
        assert beta > 0

        idxes = self._sample_proportional(batch_size,_it_sum)

        weights = []
        p_min = _it_min[0].min() / max(_it_sum[0].sum(),1e-5)
        max_weight = (p_min * len(self._storage)+1e-5) ** (-beta)

        for idx in idxes:
            p_sample = _it_sum[0][idx] / max(_it_sum[0].sum(),1e-5)
            weight = (p_sample * len(self._storage)+1e-5) ** (-beta)
            weights.append(weight / max_weight)
        #print self._it_min.min(), weights
        weights = np.array(weights)
        weights /= np.sum(weights)
        ret = []
        for i in xrange(batch_size):
            # print(111111111111111111111111111111)
            # print(np.shape(idxes))
            # print(i)
            # print(idxes[i])
            # print(np.shape(self._storage))
            # print(111111111111111111111111111111)
            ret.append(self._storage[idxes[i]])
        return (ret, idxes, weights)

    def update_priorities(self, idxes, priorities,_it_sum,_it_min):
        assert len(idxes) == len(priorities)
        #print priorities, np.sum(priorities)
        for idx, priority in zip(idxes, priorities):
            #print priority
            assert priority >= 0, "priority is "+str(priority)
            assert 0 <= idx < len(self._storage)
            
            _it_sum[0][idx] = priority ** self._alpha
            _it_min[0][idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

