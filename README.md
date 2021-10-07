# MultiAgentVariationalOcclusionInference
Multi-agent occlusion inference using observed driver behaviors. A driver sensor model is learned using a conditional variational autoencoder which maps an observed driver trajectory to the space ahead of the driver, represented as an occupancy grid map (OGM). Information from multiple drivers is fused into an ego vehicle's map using evidential theory. See our [video](https://www.youtube.com/watch?v=cTHl5nDBNBM) and [paper](https://arxiv.org/abs/2109.02173) for more details:

M. Itkina, Y.-J. Mun, K. Driggs-Campbell, and M. J. Kochenderfer. "Multi-Agent Variational Occlusion Inference Using People as Sensors". ArXiv, 2021.

![image](https://user-images.githubusercontent.com/24766091/132141370-373c073e-bc24-4482-911f-32d3f9581ff0.png)
**Figure:** Our proposed occlusion inference approach. The learned driver sensor model maps behaviors of visible drivers (cyan) to an OGM of the environment ahead of them (gray). The inferred OGMs are then fused into the ego vehicle’s (green) map. Our goal is to infer the presence or absence of occluded agents (blue). Driver 1 is waiting to turn left, occluding oncoming traffic from the ego vehicle’s view. Driver 1 being stopped should indicate to the ego that there may be oncoming traffic; and thus it is not safe to proceed with its right turn. Driver 2 is driving at a constant speed. This observed behavior is not enough to discern whether the vehicle is traveling in traffic or on an open road. We aim to encode such intuition into our occlusion inference algorithm.

## Instructions
The code reproduces the qualitative and quantitative experiments in the paper. The required dependencies are listed in dependencies.txt. Note that the INTERACTION dataset for the GL intersection has to be downloaded from: https://interaction-dataset.com/ and placed into the `data` directory. Then, a directory: `data/INTERACTION-Dataset-DR-v1_1` should exist.

To run the experiments, please run the following files:

- To process the data:
`scripts/run_preprocess_data.sh`.

- To train and test the driver sensor models (ours, k-means PaS, GMM PaS):
`scripts/train_and_test_driver_sensor_model.sh`.

- To run and evaluate the multi-agent occlusion inference pipeline:
`scripts/run_full_pipeline.sh`.

Running the above code will reproduce the numbers in the tables reported in the paper. To visualize the qualitative results, see `src/driver_sensor_model/visualize_cvae.ipynb` (Fig. 3) and `src/full_pipeline/main_visualize_full_pipeline.py` (Figs. 4 and 10).

## Data Split
To process the data, ego vehicle IDs were sampled from the GL intersection in the [INTERACTION dataset](https://interaction-dataset.com/). For each of the 60 available scenes, a maximum of 100 ego vehicles were chosen. The train, validation, and test set were randomly split based on the ego vehicle IDs accounting for 85%, 5% and 10% of the total number of ego vehicles, respectively. Due to computational constraints, the training set for the driver sensor model was further reduced to have 70,001 contiguous driver sensor trajectories or 2,602,332 time steps of data. The validation set used to select the driver sensor model and the sensor fusion scheme contained 4,858 contiguous driver sensor trajectories or 180,244 time steps and 289 ego vehicles. The results presented in this paper were reported on the test set, which consists of 9,884 contiguous driver sensor trajectories or 365,201 time steps and 578 ego vehicles.

## CVAE Driver Sensor Model Architecture and Training
We set the number of latent classes in the CVAE to K = 100 based on computational time and tractability for the considered baselines. We standardize the trajectory data to have zero mean and unit standard deviation. The prior encoder in the model consists of an LSTM with a hidden dimension of 5 to process the 1 s of trajectory input data. A linear layer then coverts the output into a K-dimensional vector that goes into a softmax function, producing the prior distribution. The posterior encoder extracts features from the ground truth input OGM using a [VQ-VAE](https://arxiv.org/abs/1711.00937) backbone with a hidden dimension of 4. These features are flattened and concatenated with the LSTM output from the trajectory data, and then passed into a linear layer and a softmax function, producing the posterior distribution. The decoder passes the latent encoding through two linear layers with ReLU activation functions and a transposed VQ-VAE backbone, outputting the inferred OGM.

To avoid latent space collapse, we clamped the KL divergence term in the loss in at 0.2. Additionally, we anneal the beta hyperparameter in the loss according to a sigmoid schedule. In our hyperparameter search, we found a maximum beta of 1 with a crossover point at 10,000 iterations to work well. The beta hyperparameter increases from a value of 0 to 1 over 1,000 iterations. We set the alpha hyperparameter to 1.5. We trained the network with a batch size of 256 for 30 epochs using the [Adam optimizer](https://arxiv.org/abs/1412.6980) with a starting learning rate of 0.001.
