# Neural Network (NN) Graphical Represenations


The graphical information represented below displays current testing values of four different Convolutional Neural Networks as listed:
- Autoencoder
  - Standard encoder/decoder setup.
- VAE
  - Variational Autoencoder
- $\beta$-VAE
  - Variational Autoencoder with a $\beta=5$
- Donkey Car
  - NN architecture taken from the Donkey Car project for comparison purposes
  - A TensorFlow implementation of the network from https://github.com/wroscoe/donkey

---

## Correlations
- Plots display ground truth steering command (x-axis) against the value of each individual element of the embedding vector (y-axis) for each image in the test set. The aim was to visualize any corrilation between steering command and 1 or more elements of the embedding vector. However, none are apparent:
#### Autoencoder
![Autoencoder Correlations](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/autoencoder/correlations.png)
#### VAE
![VAE Correlation](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/vae/correlations.png)
#### b(5)-VAE
![b5-VAE Correlations](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/vae/correlations.png)

## Side-by-side
- Noisy input image to the networks (top). Reconstructed image (bottom):
#### Autoencoder
![Autoencoder Side-by-side](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/autoencoder/side_by_side.jpg)
#### VAE
![VAE Side-by-side](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/vae/side_by_side.jpg)
#### b(5)-VAE
![b5-VAE Side-by-side](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/beta_5_vae/side_by_side.jpg)

## Visualizations
- Taking the average embedding across the entire test set as a starting point (center column). Each element of the 50 element embedding vector is swepped +- 6 from the mean with a step of 1, while holding all other element constan.:
#### Autoencoder
![Autoencoder Visualizations](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/autoencoder/visualizations.png)
#### VAE
![VAE Visualizations](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/beta_5_vae/visualizations.png)
#### b(5)-VAE
![b5-VAE Visualizations](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/vae/visualizations.png)

---

## Steering Scatter Plot
- Plots display the correlation between expected steering position (annotations) and positions predicted by the Neural Net:
#### Donkey Car
![Donkey Car Steering Scatter](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/donkey_car_tetsing/donkey_steeing_scatter.png)
#### Autoencoder
![Autoencoder Steering Scatter](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_steering/autoencoder/Steering_scatter.png)
#### VAE
![VAE Steering Scatter](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_steering/vae/Steering_scatter.png)
#### b(5)-VAE
![b5-VAE Steering Scatter](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_steering/beta_5_vae/Steering_scatter.png)

## Steering v Time
- Graphs show overlaid steering positions (steering command); from full left to full right; as taken from the data set (Ground Truth) and the Neural Net (Predictions):
- Error indicates deviation by the predicted values from the ground truth:
#### Donkey Car
![Donkey Car Steering v Time](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/donkey_car_tetsing/donkey_steering_v_time.png)
#### Autoencoder
![Autoencoder Steering v Time](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_steering/autoencoder/Steering_v_time.png)
#### VAE
![VAE Steering v Time](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_steering/vae/Steering_v_time.png)
#### b(5)-VAE
![b5-VAE Steering v Time](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_steering/beta_5_vae/Steering_v_time.png)

## Throttle Scatter Plot
- Plots display the correlation between expected throttle value (annotations) and positions predicted by the Neural Net:
#### Donkey Car
![Donkey Car Throttle Scatter](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/donkey_car_tetsing/donkey_throttle_scatter.png)
#### Autoencoder
![Autoencoder Throttle Scatter](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_throttle/autoencoder/Throttle_scatter.png)
#### VAE
![VAE Throttle Scatter](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_throttle/vae/Throttle_scatter.png)
#### b(5)-VAE
![b5-VAE Throttle Scatter](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_throttle/beta_5_vae/Throttle_scatter.png)

## Throttle v Time
- Graphs show throttle values over predicted values with respect to time:
- Error indicates deviation by the predicted values from the ground truth:
#### Donkey Car
![Donkey Car Throttle v Time](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/donkey_car_tetsing/donkey_throttle_v_time.png)
#### Autoencoder
![Autoencoder Throttle v Time](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_throttle/autoencoder/Throttle_v_time.png)
#### VAE
![VAE Throttle v Time](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_throttle/vae/Throttle_v_time.png)
#### b(5)-VAE
![b5-VAE Throttle v Time](https://github.com/tall-josh/fyp_diy_robo_car/blob/master/report/modular_throttle/beta_5_vae/Throttle_v_time.png)
