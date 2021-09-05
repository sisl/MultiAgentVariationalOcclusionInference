cd ../src/full_pipeline
python main_save_full_pipeline.py  --model=vae --mode=evidential
python main_save_full_pipeline.py  --model=gmm --mode=evidential
python main_save_full_pipeline.py  --model=kmeans --mode=evidential
python main_save_full_pipeline.py  --model=vae --mode=average

# Change the model and fusion mode in the file to obtain metrics.
python full_pipeline_metrics.py

# Visualize the scenarios in the paper and in the appendix.
python main_visualize_full_pipeline.py --model=vae --mode=evidential

