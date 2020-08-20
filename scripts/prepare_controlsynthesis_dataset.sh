
#generate training data for synthesis model

mkdir -p ./datasets/$1/synthesis_tfr

ddsp_prepare_tfrecord --input_audio_filepatterns="./datasets/$1/audio_16k/*" --output_tfrecord_path="./datasets/$1/synthesis_tfr/" --num_shards=1 --alsologtostderr 

#generate non overlapping data to be turned into data for control model

mkdir -p ./datasets/$1/pre_control_tfr

ddsp_prepare_tfrecord --input_audio_filepatterns="./datasets/$1/audio_16k/*" --output_tfrecord_path="./datasets/$1/pre_control_tfr/" --num_shards=1 --alsologtostderr --example_secs=4 --sliding_window_hop_secs=4


#augment control model data with midi transcriptions.
mkdir -p ./datasets/$1/control_dataset

python3 create_control_model_dataset.py --input_tfr_path ./datasets/$1/pre_control_tfr --output_tfr_path ./datasets/$1/control_dataset