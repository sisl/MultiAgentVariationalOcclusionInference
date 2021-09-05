# MultiAgentVariationalOcclusionInference
Multi-agent occlusion inference using observed driver behaviors. A driver sensor model is learned using a conditional variational autoencoder which maps an observed driver trajectory to the space ahead of the driver, represented as an occupancy grid map (OGM). Information from multiple drivers is fused into an ego vehicle's map using evidential theory. See our paper for more details: "Multi-Agent Variational Occlusion Inference Using People as Sensors". 

The code reproduces the qualitative and quantitative experiments in the paper. The required dependencies are listed in dependencies.txt. Note that the INTERACTION dataset for the GL intersection has to be downloaded from: https://interaction-dataset.com/ and placed into the data directory. Then, a directory: data/INTERACTION-Dataset-DR-v1_1 should exist.

To run the experiments, please run the following files:

To process the data:
scripts/run_preprocess_data.sh

To train and test the driver sensor models (ours, k-means PaS, GMM PaS):
scripts/train_and_test_driver_sensor_model.sh

To run and evaluate the multi-agent occlusion inference pipeline:
scripts/run_full_pipeline.sh

Running the above code will reproduce the numbers in the tables reported in the paper. To visualize the qualitative results, see src/driver_sensor_model/visualize_cvae.ipynb (Fig. 3) and src/full_pipeline/main_visualize_full_pipeline.py (Figs. 4 and 10).
