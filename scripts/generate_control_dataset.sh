mkdir ./datasets/$1/control_dataset

python3 create_control_model_dataset.py --input_tfr_path ./datasets/$1/pre_control_tfr --output_tfr_path ./datasets/$1/control_dataset 