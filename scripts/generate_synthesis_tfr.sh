#!/bin/bash

mkdir -p ./datasets/$1/synthesis_tfr
echo $2

ddsp_prepare_tfrecord --input_audio_filepatterns="./datasets/$1/audio_16k/*" --output_tfrecord_path="./datasets/$1/synthesis_tfr/$1_tfr" --num_shards=1 --alsologtostderr --example_secs=$2 --example_secs=$2 --sliding_window_hop_secs=$2