# Deep Q-Network (DQN) implemeted by PyTorch for the Unity-based Banana Envirnment
---

This repository is an implementation of the DQN algorithm for the Banana Environment developed by Unity3D and accessed through the UnityEnvironment library. It is an extension of the code sample provided by the Udacity Deep RL teaching crew (for more information visit their [website](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)). The environment is presented as a vector; thus, we did not use Convolutional Neural Networks (CNN) in the implementation.

This repository consists of these files:

*These files are saved under the "src" directory.*
1. <ins> model.py </ins>: This module provides the underlying neural network for our agent. When we train our agent, this neural network is going to be updated by backpropagation.
2. <ins>buffer.py</ins>: This module implements the "memory" of our agent, also known as the Experience Replay.
3. <ins>agent.py</ins>: This is the body of our agent. It implements the way the agent acts (using $$\epsilon$$-greedy policy), and learn an optimal policy.
4. <ins>train.py</ins>: This module has the train function which takes the agent, the environment, number of training episodes and the required hyper-parameters and trains the agent accordingly.

To test the code, after cloning the project, open the `Navigation.ipynb` notebook. It has all the necessary steps to install and load the packages, and train and test the agent. There is an already trained agent stored in `checkpoint.pth`, by running the last part of the notebook, this can be directly tested.
