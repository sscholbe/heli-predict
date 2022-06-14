# Neural Network Trajectory Prediction
This is a machine learning project I tinkered with for the past year. The goal is to increase the accuracy of an aimbot for helicopters in Battlefield 4.

Some disclaimers at first:
- This project is solely for educational purposes and I don't endorse cheating.
- I won't provide any source code.
- I won't discuss reverse engineering of the engine, anti-cheat bypasses, etc.

## Why a neural network?
Conventional prediction methods...

## Goals
- High accuracy (± 1 m) over long distances (300 - 1000 m) and bullet travel times up to 1 second.
- Smooth and consistent predictions (i.e. no jitter when aiming).
- High responsiveness to changes of movement (e.g. detect sudden acceleration and adjust predictions)

## Technical challenges
- Low network inference time due to server updates at 60 Hz.
- Low computational complexity due to high CPU/GPU utilization by the game itself.
- Small footprint in memory and code to minimize the risk of detection by the Anti-Cheat system.

## Data collection
I collected the data by flying around for approximately an hour on a local server and recording my helicopter's state at every frame using my software. This includes time, position, rotation and velocity. This data is provided by the game engine for all vehicles as it is required to render and update the vehicles.

## Data preparation
I preprocess the data by linearly interpolating all recorded values (i.e. position, rotation and velocity vectors) over the time axis to obtain consistent frame data every 50 ms, since the frame rate sometimes drops or spikes. I then traverse with a window over the data and create (X, Y) pairs where the given window consists of 20 frames (i.e. 1 second) containing position, rotation and velocity and the prediction window consists of 20 frames containing only the position.

An example of such data looks as follows. The blue line shows the recorded position change over the past second and the red one the future position change (i.e. what we want to predict).

