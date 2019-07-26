# Hierarchical Neuroevolution for Reinforcement Learning Tasks

This is a repository for code used for a neuroevolution approach for a reinforcement learning task.
This was a project undertaken for the Machine Learning Pratical (MLP) course at Edinburgh University by Michael Westerman and Michael Decker. This was shortlisted for the 2019 IBM Prize for the best project Machine Learning Practical.

The neuroevolution algorithm uses a hierarchical genetic algorithm to update weights of a neural network designed to output actions given state information.

An implementation of Twin Delayed Deep Deterministic policy gradient (TD3, https://arxiv.org/abs/1802.09477) is also included for comparison. This is (to my current knowledge) the current state of the art and best performing reinforcement learning algorithm when tested on MuJoCo environments. 

## Requirements
 - Python (> 3.5)
 - NumPy
 - Pytorch
 - Roboschool
 - Matplotlib
