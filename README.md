
# Neural Network Trajectory Prediction
This is a machine learning project I tinkered with for the past year. The goal is to increase the accuracy of an aimbot for helicopters in Battlefield 4.

Some disclaimers at first:
- This project is solely for educational purposes and I don't endorse cheating.
- I won't provide any source code.
- I won't discuss reverse engineering of the engine, anti-cheat bypasses, etc.

## Why a neural network?
Conventional prediction methods...

## Overview

We want our software to intercept the regular game loop (consisting of physics updates, responses to user input, rendering etc.) to record necessary information such as vehicle position, rotation and velocity of enemy helicopters. We then feed this information into our network to obtain a prediction. We can use the prediction to automatically adjust the user's aim based on bullet speed/drop and distance to the enemy helicopter.

![overview](https://user-images.githubusercontent.com/79590619/173553307-e212fe6d-dd82-415e-a120-e70a2e524bb3.png)

## Goals
- High accuracy (± 1 m) over long distances (300 - 1000 m) and bullet travel times up to 1 second.
- Smooth and consistent predictions (i.e. no jitter when aiming).
- High responsiveness to sudden changes of movement (e.g. detect strong acceleration and adjust predictions quickly)

## Technical challenges
- Low network inference time due to server updates at 60 Hz.
- Low computational complexity due to high CPU/GPU utilization by the game itself.
- Small footprint in memory and code to minimize the risk of detection by the Anti-Cheat system.

## Data collection
We collected the data by flying around for approximately an hour on a local server and recording the helicopter's state at every frame using a software. This includes time, position, rotation and velocity. This data is provided by the game engine for all vehicles as it is required to render and update the vehicles.

## Data preparation
We preprocess the data by linearly interpolating all recorded values (i.e. position, rotation and velocity vectors) over the time axis to obtain consistent frame data every 50 ms, since the frame rate sometimes drops or spikes. We then traverse with a window over the data and create (X, Y) pairs where the given window consists of 20 frames (i.e. 1 second) containing position, rotation and velocity and the prediction window consists of 20 frames containing only the position.

An example of such data looks as follows. The blue line shows the recorded position change (X/Y/Z axis separately) over the past second and the red one the future position change (i.e. what we want to predict).

![example_data](https://user-images.githubusercontent.com/79590619/173551848-34deb616-4198-4816-8772-2bd2124ff18d.png)

## Network architecture

We choose a polynomial of degree two per axis as our model to describe the future movement of the helicopter. This seems sufficient for most short term movement. By centering the data around time 0, we can drop the intercept to guarantee continuity. The network thus determines two coefficients per dimension, i.e. a total of six coefficients. Since the evaluation of polynomials is linear, we can add a layer to our network that computes the prediction for each future timestep using a simple matrix multiplication.

$$p_{pred}(t)=p_0+\left(
\begin{array}{c}
x_1\\
y_1\\
z_1
\end{array}
\right)(t-t_0)+\left(
\begin{array}{c}
x_2\\
y_2\\
z_2
\end{array}
\right)(t-t_0)^2
$$

The network itself consists of three smaller sub-networks that predict two coefficients each (i.e. per axis). They consist of two fully connected layers, a skip connection from the input to the second layer and an ELU activation.

![network](https://user-images.githubusercontent.com/79590619/173553347-a812efc5-1f65-4b7e-9667-04958d857d87.png)

## Training

We train our network over 100 epochs with a batch size of 32 on shuffled data. As our loss, we use the mean squared error between our predictions (i.e. the evaluated polynomial) and the ground truth. On Google Colab, this takes around three to four minutes.

## Results



