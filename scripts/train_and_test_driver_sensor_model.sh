cd ../src/driver_sensor_model
# Train and test our CVAE driver sensor model.
python train_cvae.py --norm --mut_info='const' --epochs=30 --learning_rate=0.001 --beta=1.0 --alpha=1.5 --latent_size=100 --batch_size=256 --crossover=10000
python inference_cvae.py --norm --mut_info='const' --epochs=30 --learning_rate=0.001 --beta=1.0 --alpha=1.5 --latent_size=100 --batch_size=256 --crossover=10000

# Train and test the k-means PaS baseline driver sensor model.
python train_kmeans.py
python inference_kmeans.py

# Train and test the GMM PaS baseline driver sensor model.
python train_gmm.py
python inference_gmm.py

# To view the qualitative driver sensor model results, please see the Jupyter notebook: src/driver_sensor_model/visualize_cvae.ipynb.
