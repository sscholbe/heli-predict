# Neural Network Trajectory Prediction for Helicopters in Battlefield 4
This is a machine learning side project I have worked on from time to time for the past few months. The goal is to increase the accuracy of an aimbot against helicopters in Battlefield 4 using neural networks. 

Some disclaimers at first:
- This project is solely for educational purposes, and I do not endorse cheating.
- I do not provide any source code.
- I do not discuss reverse engineering of the engine, anti-cheat bypasses, etc.

## Why a neural network?

In Battlefield 4, helicopters are very agile and bullets have ballistics. We usually have to aim quite far ahead from the location on the player screen to hit an enemy. Since the helicopters are agile and thus hard to predict, programmately hitting helicopters using an aimbot is challenging.

![trajectory](https://user-images.githubusercontent.com/79590619/173610035-ddc35520-0058-4c15-a5ff-2a83794ec95c.png)

While the bullet ballistics can be well predicted, the helicopter trajectory cannot. The standard approach is to do **linear extrapolation** to predict the future location of such vehicles. That means we take the direction or velocity at the current frame to add it to the current position. For example, if we, at the current time, drive north with a speed of 60 km/h, we will likely have moved 1 km north in the next minute (although we could have braked, turned, etc.).

![linear_extrapolation](https://user-images.githubusercontent.com/79590619/173618459-82ad9475-f0f7-49c4-ab80-71c232e91c5b.png)

This approach works sufficiently for slower and larger vehicles such as tanks. However, linear extrapolation usually drastically under and overestimates the movement of helicopters. That means low accuracy, especially on longer ranges (i. e., longer bullet times).

Other approaches, such as fitting a higher degree polynomial on the past data to extrapolate, do not work well since they have slow responsiveness, and historic motions usually do not truly reflect future motions. That makes it an excellent machine learning challenge.

## Overview

We want our software to intercept the regular game loop (consisting of physics updates, responses to user input, rendering, etc.) to record necessary information such as vehicle position, rotation, and velocity of enemy helicopters. We then feed this information into our network to obtain a prediction. We use the prediction to automatically adjust the user's aim based on bullet speed/drop and distance. Since we do not discuss reverse engineering, we simply assume everything that needs to be done can be done, and we will only focus on the neural network part.

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
We collect the data by flying around on a local server and recording the helicopter's state at every frame using the software (~35 min of footage). This includes time, position, rotation, and velocity. The game engine provides this data for all vehicles as it is required to render and update them.

The position data of the test set (~5 min of footage) looks as follows:

![test_set](https://user-images.githubusercontent.com/79590619/173807357-bae8b8a2-6399-460b-af17-f721ab287d77.png)

## Data preparation
We preprocess the data by linearly interpolating all recorded values (i.e., position, rotation, and velocity vectors) over the time axis to obtain consistent frame data every 50 ms since the frame rate sometimes drops or spikes. We then traverse with a window over the data and create (X, Y) pairs where the given window consists of 20 frames (i.e., 1 second) containing position, rotation, and velocity. The prediction window consists of 20 frames containing only the position.

An example of such data looks as follows. The blue line shows the recorded position changes over the last second (i.e., what is given to the neural network), and the red one shows the future position changes (i.e., what we want to predict).

![example_data](https://user-images.githubusercontent.com/79590619/173808325-0cb1c5e1-341c-4443-ad7c-7ba10105da0b.png)

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

We train our network over 100 epochs with a batch size of 32 on shuffled data using the NAdam optimizer. As our loss, we use the mean squared error between our predictions (i.e., the evaluated polynomial) and the ground truth.

## Results

We measure a **significant increase in accuracy (2.07x)** with our neural network (**NN**) compared to the old method using linear extrapolation (**LE**). We also see that the smallest hitbox (when looking from the front onto the helicopter) is covered almost entirely by the mean in our method. Since a bullet either hits the hitbox (and does damage) or doesn't, this clearly **increases our average hit rate from 56% to 82%**.

![prediction_error](https://user-images.githubusercontent.com/79590619/173809531-1d0f44ae-b1e3-43bb-9d85-5646ead8f354.png)

We also note a larger prediction error on the Y-axis (up/down) for both methods, which we assume is due to the up/down movements being much more instantaneous in the game engine. However, our method **increases accuracy on the Y-axis by 1.75x and by 2.38x on the XZ-plane**.

![prediction_error_xyz](https://user-images.githubusercontent.com/79590619/173811595-44326de0-ff56-4f37-9116-3ccfc475dcff.png)

## Application in practice

With the trajectory prediction working, we can create an aimbot by finding the intersection between a bullet trajectory (which can be well approximated using a polynomial of degree two) and the predicted helicopter trajectory polynomial. We then simulate user input to aim at the correct location and shoot. The following animations show the aimbot in action.
