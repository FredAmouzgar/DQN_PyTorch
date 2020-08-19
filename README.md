# Deep Q-Network (DQN) implemeted by PyTorch for the Unity-based Banana Envirnment
---

This repository is an implementation of the DQN algorithm for the Banana Environment developed by Unity3D and accessed through the UnityEnvironment library. It is an extension of the code sample provided by the Udacity Deep RL teaching crew. The environment is presented as a vector; thus, we did not use Convolutional Neural Networks (CNN) in the implementation.

This repository consists of these files:
Main files are saved under the "src" directory.
1- model.py: This module provides the underlying neural network for our agent. When we train our agent, this neural network is going to be updated by backpropagation.
2- buffer.py: This module implements the "memory" of our agent, also known as the Experience Replay.