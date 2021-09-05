# MultiAgentVariationalOcclusionInference
Multi-agent occlusion inference using observed driver behaviors. A driver sensor model is learned using a conditional variational autoencoder which maps an observed driver trajectory to the space ahead of the driver, represented as an occupancy grid map (OGM). Information from multiple drivers is fused into an ego vehicle's map using evidential theory. See our paper for more details: 

M. Itkina, Y.-J. Mun, K. Driggs-Campbell, and M. J. Kochenderfer. "Multi-Agent Variational Occlusion Inference Using People as Sensors". ArXiv, 2021.

The code reproduces the qualitative and quantitative experiments in the paper. The required dependencies are listed in dependencies.txt. Note that the INTERACTION dataset for the GL intersection has to be downloaded from: https://interaction-dataset.com/ and placed into the data directory. Then, a directory: `data/INTERACTION-Dataset-DR-v1_1` should exist.
![image](https://user-images.githubusercontent.com/24766091/132141370-373c073e-bc24-4482-911f-32d3f9581ff0.png)
Our proposed occlusion inference approach. The learned driver sensor model maps behaviors of visible drivers (cyan) to an OGM of the environment ahead of them (gray). The inferred OGMs are then fused into the ego vehicle’s (green) map. Our goal is to infer the presence or absence of occluded agents (blue). Driver 1 is waiting to turn left, occluding oncoming traffic from the ego vehicle’s view. Driver 1 being stopped should indicate to the ego that there may be oncoming traffic; and thus it is not safe to proceed with its right turn. Driver 2 is driving at a constant speed. This observed behavior is not enough to discern whether the vehicle is traveling in traffic or on an open road. We aim to encode such intuition into our occlusion inference algorithm.

To run the experiments, please run the following files:

- To process the data:
`scripts/run_preprocess_data.sh`.

- To train and test the driver sensor models (ours, k-means PaS, GMM PaS):
`scripts/train_and_test_driver_sensor_model.sh`.

- To run and evaluate the multi-agent occlusion inference pipeline:
`scripts/run_full_pipeline.sh`.

Running the above code will reproduce the numbers in the tables reported in the paper. To visualize the qualitative results, see `src/driver_sensor_model/visualize_cvae.ipynb` (Fig. 3) and `src/full_pipeline/main_visualize_full_pipeline.py` (Figs. 4 and 10).
