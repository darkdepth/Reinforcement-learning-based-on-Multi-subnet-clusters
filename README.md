# Reinforcement-learning-based-on-Multi-subnet-clusters(MSC-RL):

A reinforcement learning algorithm based on multi-subnet cluster(MSC-RL).
Suspected will never overfit :)

### Abstract:
We proposed a reinforcement learning algorithm based on multi-subnet cluster(MSC-RL). The proposed network consists of multiple subnet clusters and the primary storage network. Each subnet cluster is composed of multiple subnets and one sub-storage network. 

In the subnet cluster, multiple subnets are used to explore the solution space simultaneously and saves the searched information to the sub-storage network. At regular intervals, the subnet cluster saves the searched information to the primary storage network. 

MSC-RL can exchange information searched by each subnet through the sub-storage network to realize information interaction within the subnet cluster. Each cluster uses the primary storage network for information interaction. The method enhances the information interaction between subnets and improves the ability of the algorithm to optimize. 

### Experiments:
This article uses Pong in Atari as the experimental environment.

The proposed algorithm obtains 21.0. It is the maximum score that can be obtained in the game environment. That is to say, the maximum score that can be obtained by the proposed method. The results showed that our method can find the optimal value of the current environment. 

### Installation Dependencies:
- Python 2.7 or 3
- TensorFlow 1.0+
- Pygame
- Numpy

### How to run:
    python myapexdqn_3_8.py
