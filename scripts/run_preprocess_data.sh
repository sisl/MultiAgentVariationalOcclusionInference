# Record the mean and standard deviation statistics at the end into statistics.txt and into src/utils/data_generator.py in three functions.
cd ../src/preprocess
python generate_data.py
python train_val_test_split.py
python get_driver_sensor_data.py
python preprocess_driver_sensor_data.py
