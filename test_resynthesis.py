from ddsp.colab.colab_utils import play, specplot
import ddsp.training
import gin
from matplotlib import pyplot as plt
import numpy as np
import librosa
import tensorflow as tf
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument('--dataset')

args = parser.parse_args()

DATASET_NAME = args.dataset

SAMPLE_RATE = 16000

DEMO_DURATION = 30

MIDI_FRAME_RATE = 250

TRAIN_TFRECORD_FILEPATTERN = f"./datasets/{DATASET_NAME}/synthesis_tfr/*"

synthesis_model_checkpoint_path = f"./datasets/{DATASET_NAME}/synthesis_checkpoints"

data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)

dataset = data_provider.get_batch(batch_size=1, shuffle=False)

try:
    batch = next(iter(dataset))
except OutOfRangeError:
    raise ValueError(
        'TFRecord contains no examples. Please try re-running the pipeline with '
        'different audio file(s).')

# Parse the gin config.
gin_file = os.path.join(synthesis_model_checkpoint_path,
                        'operative_config-0.gin')
gin.parse_config_file(gin_file)


# Load model
model = ddsp.training.models.Autoencoder()
model.restore(synthesis_model_checkpoint_path)

# Resynthesize audio.
audio_gen = model(batch, training=False)
audio = batch['audio']


SAMPLE_RATE = 16000


librosa.output.write_wav(
    f"rstests/{DATASET_NAME}_rc.wav", audio_gen.numpy().reshape(-1, 1), SAMPLE_RATE, norm=True)

librosa.output.write_wav(
    f"rstests/{DATASET_NAME}.wav", audio.numpy().reshape(-1, 1), SAMPLE_RATE, norm=True)
