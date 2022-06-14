# Neural Network Trajectory Prediction for Battlefield 4
This is a machine learning side project I have worked on from time to time for the past few months. The goal is to increase the accuracy of an aimbot against helicopters in Battlefield 4 using neural networks. 

Some disclaimers at first:
- This project is solely for educational purposes, and I do not endorse cheating.
- I will not provide any source code.
- I will not discuss reverse engineering of the engine, anti-cheat bypasses, etc.

## Why a neural network?

In Battlefield 4, helicopters are very agile, and bullets have ballistics, making hitting helicopters using an aimbot quite challenging. While the bullet ballistics can be well predicted (using polynomials), the helicopter trajectory cannot. The standard approach is to do linear extrapolation to predict the future location of such vehicles. That means we take the direction or velocity at the current frame to add it to the current position.

This approach works sufficiently for slower and larger vehicles such as tanks. However, linear extrapolation usually drastically under and overestimates the movement of helicopters. That means low accuracy, especially on longer ranges (i. e., longer bullet times).

![trajectory](https://user-images.githubusercontent.com/79590619/173610035-ddc35520-0058-4c15-a5ff-2a83794ec95c.png)

## Overview

We want our software to intercept the regular game loop (consisting of physics updates, responses to user input, rendering, etc.) to record necessary information such as vehicle position, rotation, and velocity of enemy helicopters. We then feed this information into our network to obtain a prediction. We use the prediction to automatically adjust the user's aim based on bullet speed/drop and distance to the enemy helicopter.

![overview](https://user-images.githubusercontent.com/79590619/173553307-e212fe6d-dd82-415e-a120-e70a2e524bb3.png)

## Goals
- High accuracy (± 3 m) over long distances (300 - 1000 m) and bullet travel times up to 1 second.
- Smooth and consistent predictions (i.e., no jitter when aiming).
- High responsiveness to sudden changes of movement (e.g., detect strong acceleration and adjust predictions quickly)

## Technical challenges
- Low network inference time due to server updates at 60 Hz.
- Low computational complexity due to the game's high CPU/GPU utilization.
- Small footprint in memory and code to minimize the risk of detection by the Anti-Cheat system.

## Data collection
We collected the data by flying around for approximately an hour on a local server and recording the helicopter's state at every frame using the software. This includes time, position, rotation, and velocity. The game engine provides this data for all vehicles as it is required to render and update them.

## Data preparation
We preprocess the data by linearly interpolating all recorded values (i.e., position, rotation, and velocity vectors) over the time axis to obtain consistent frame data every 50 ms since the frame rate sometimes drops or spikes. We then traverse with a window over the data and create (X, Y) pairs where the given window consists of 20 frames (i.e., 1 second) containing position, rotation, and velocity. The prediction window consists of 20 frames containing only the position.

An example of such data looks as follows. The blue line shows the recorded position change (X/Y/Z axis separately) over the last second, and the red one shows the future position change (i.e., what we want to predict).

![example_data](https://user-images.githubusercontent.com/79590619/173551848-34deb616-4198-4816-8772-2bd2124ff18d.png)

## Network architecture

We choose a polynomial of degree two per axis as our model to describe the future movement of the helicopter. This seems sufficient for most short-term movements. We can drop the intercept by centering the data around time 0 to guarantee continuity. The network thus determines two coefficients per dimension, i.e., six coefficients.

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

Since the evaluation of polynomials is linear, we can add a layer to our network that computes the prediction for each future timestep using simple matrix multiplication. The network consists of three smaller sub-networks that predict two coefficients each (i.e., per axis). Each consists of two fully connected layers, a skip connection from the input to the second layer and an ELU activation.

![network](https://user-images.githubusercontent.com/79590619/173553347-a812efc5-1f65-4b7e-9667-04958d857d87.png)

## Training

We train our network over 100 epochs with a batch size of 32 on shuffled data using the Adam optimizer. As our loss, we use the mean squared error between our predictions (i.e., the evaluated polynomial) and the ground truth.

## Results

We measure a **significant increase in accuracy (2.09x)** with our neural network compared to the old method using linear extrapolation. Our new method increases the frontal accuracy range (where the helicopter is looking at the player; thus, the hitbox is smaller) from 0.35 s to 0.55 s and the sideways accuracy range (where the hitbox is longer due to the tail) from 0.65 to 0.95 s.

![prediction_error](https://user-images.githubusercontent.com/79590619/173598826-9e08ea87-1fbc-4b1f-83ec-b95414e94dba.png)

## Application in practice

With the trajectory prediction working, we can create an aimbot by finding the intersection between a bullet trajectory (which can be well approximated using a polynomial of degree two) and the predicted helicopter trajectory polynomial. We then simulate user input to aim at the correct location and shoot. The following animations show the aimbot in action.
