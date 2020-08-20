from data import MidiDataTFRecordProvider, NpyDataProvider
from preprocessing import MidiPreprocessor
from ddsp.colab import colab_utils
import ddsp
from matplotlib import pyplot as plt
import numpy as np
import os
from control_models import AimuControlModel
import tensorflow.compat.v2 as tf
import tensorboard as tb
from absl import logging
import time
from trn_lib import tr, save, restore, plot_conditionings, generate_audio_examples, save_audio_from_dict, write_audio_dict_to_summary
from synthesis_models import get_trained_synthesis_model
from load_midi import load_midi, load_midi_examples
from plot_dataset_stats import plot_dataset_stats
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--ds_id')

args = parser.parse_args()

DATASET_NAME = args.ds_id

CLIP_DURATION = 16

SAMPLE_RATE = 16000

FEATURE_FRAME_RATE = 250


control_model = AimuControlModel(
    n_timesteps=CLIP_DURATION*FEATURE_FRAME_RATE)

EXPERIMENT_ID = "0"

BATCH_SIZE = 4

EXAMPLE_RENDER_INTERVAL = 50

PLOT_DS_STATS = True


experiment_name = f"{DATASET_NAME}_{type(control_model).__name__}_{EXPERIMENT_ID}"

synthesis_model_checkpoint_path = f"./datasets/{DATASET_NAME}/synthesis_checkpoints"

train_tfrecord_filepattern = f"./datasets/{DATASET_NAME}/control_dataset/trn"

test_tfrecord_filepattern = f"./datasets/{DATASET_NAME}/control_dataset/tst"

checkpoint_dir = f"./experiments/{experiment_name}/checkpoints"
summary_dir = f"./experiments/{experiment_name}/summaries"

audio_example_path = f"./experiments/{experiment_name}/audio_examples"


for d in [checkpoint_dir, summary_dir, audio_example_path]:
    if not os.path.exists(d):
        os.makedirs(d)


tf.random.set_seed(
    1
)

try:

    trn_data_provider = NpyDataProvider(
        train_tfrecord_filepattern)

    tst_data_provider = NpyDataProvider(
        test_tfrecord_filepattern)

except Exception as e:

    print(e)

    trn_data_provider = MidiDataTFRecordProvider(
        train_tfrecord_filepattern)

    tst_data_provider = MidiDataTFRecordProvider(
        test_tfrecord_filepattern)

if PLOT_DS_STATS:
    plot_dataset_stats(trn_data_provider.get_dataset(),
                       FEATURE_FRAME_RATE, f"./datasets/{DATASET_NAME}/info")


train_sample = next(iter(trn_data_provider.get_dataset().take(1)))

n_samples = train_sample["audio"].shape[0]

clip_duration = n_samples/SAMPLE_RATE


trn_summary_writer = tf.summary.create_file_writer(summary_dir+"/trn")
tst_summary_writer = tf.summary.create_file_writer(summary_dir+"/tst")

tb.notebook.start('--logdir "{}"'.format(summary_dir))

optimizer = tf.keras.optimizers.Adam(3e-4)

# for demo

midi_demo_examples = load_midi_examples(
    midi_dir="./midi", midi_frame_rate=FEATURE_FRAME_RATE, clip_duration=CLIP_DURATION, voice_limit=1, sample_rate=SAMPLE_RATE)

tst_synthesis_model = get_trained_synthesis_model(
    synthesis_model_checkpoint_path, CLIP_DURATION=CLIP_DURATION)
demo_synthesis_model = get_trained_synthesis_model(
    synthesis_model_checkpoint_path,  CLIP_DURATION=CLIP_DURATION)

epoch = 0


epoch = restore(control_model, optimizer, epoch, checkpoint_dir)


display_batch = next(iter(tst_data_provider.get_batch(
    shuffle=False, batch_size=2, repeats=1)))


spectral_loss = ddsp.losses.SpectralLoss()


def feature_loss(model, batch, training, reconstruction_loss=True):
    out = model(batch, training=training)

    ld_loss = tf.reduce_mean(tf.keras.losses.MSE(
        out["predicted_ld_scaled"], out["ld_scaled"]))

    f0_loss = tf.reduce_mean(tf.keras.losses.MSE(
        out["predicted_f0_scaled"], out["f0_scaled"]))

    return ld_loss, f0_loss


def grad(model, batch):
    with tf.GradientTape() as tape:
        total_loss, ld_loss, f0_loss = loss(model, batch, training=True)
    return total_loss, ld_loss, f0_loss, tape.gradient(total_loss, model.trainable_variables)


while True:

    trn_batches = trn_data_provider.get_batch(
        shuffle=True, batch_size=BATCH_SIZE, repeats=1)
    tst_batches = tst_data_provider.get_batch(
        shuffle=True, batch_size=BATCH_SIZE, repeats=1)

    batch_count = 0

    epoch_ld_loss = 0
    epoch_f0_loss = 0
    epoch_total_loss = 0

    for batch in trn_batches:

        total_loss = 0
        ld_loss = 0
        f0_loss = 0

        with tf.GradientTape() as tape:

            ld_loss, f0_loss = feature_loss(
                control_model, batch, training=False)

            total_loss += ld_loss+f0_loss
            epoch_ld_loss += ld_loss
            epoch_f0_loss += f0_loss

            epoch_total_loss += total_loss

        grads = tape.gradient(
            total_loss, control_model.trainable_variables)

        optimizer.apply_gradients(
            zip(grads, control_model.trainable_variables))

        tf.print("epoch:", epoch, "step:", batch_count, "total_loss: ", tr(total_loss),
                 "ld_loss:", tr(ld_loss), "f0_loss:", tr(f0_loss))

        batch_count += 1

    with trn_summary_writer.as_default():
        tf.summary.scalar("epoch total loss:",
                          epoch_total_loss/batch_count, step=epoch)
        tf.summary.scalar("epoch ld_loss:",
                          epoch_ld_loss/batch_count, step=epoch)
        tf.summary.scalar("epoch f0_loss:", epoch_f0_loss /
                          batch_count, step=epoch)

    epoch_ld_loss = 0
    epoch_f0_loss = 0
    epoch_total_loss = 0

    batch_count = 0

    for batch in tst_batches:

        ld_loss, f0_loss = feature_loss(
            control_model, batch, training=False)

        epoch_ld_loss += ld_loss
        epoch_f0_loss += f0_loss
        epoch_total_loss += ld_loss+f0_loss

        batch_count += 1

    if epoch % EXAMPLE_RENDER_INTERVAL == 0:

        with tst_summary_writer.as_default():
            tf.summary.scalar("epoch total loss:",
                              epoch_total_loss/batch_count, step=epoch)
            tf.summary.scalar("epoch ld_loss:",
                              epoch_ld_loss/batch_count, step=epoch)
            tf.summary.scalar("epoch f0_loss:",
                              epoch_f0_loss/batch_count, step=epoch)

        # generate plot and audio examples

        tst_examples_audio = generate_audio_examples(
            control_model, tst_synthesis_model,  display_batch)
        write_audio_dict_to_summary(
            tst_examples_audio, tst_summary_writer, epoch, SAMPLE_RATE)

        save_audio_from_dict(tst_examples_audio,
                             audio_example_path, SAMPLE_RATE)

        midi_demo_audio = {}

        for midi_name, demo_batch in midi_demo_examples.items():
            demo_audio = generate_audio_examples(
                control_model, demo_synthesis_model, demo_batch, is_midi=True)
            for k in demo_audio.keys():
                midi_demo_audio[f"{midi_name}_{k}"] = demo_audio[k]

        write_audio_dict_to_summary(
            midi_demo_audio, tst_summary_writer, epoch, SAMPLE_RATE)

        save_audio_from_dict(midi_demo_audio,
                             audio_example_path, SAMPLE_RATE)

        display_performances = control_model(
            display_batch, training=False)

        plots = tf.stack(plot_conditionings(display_performances), axis=0)

        with tst_summary_writer.as_default():
            tf.summary.image("conditioning", plots, step=epoch, max_outputs=8)

        save(control_model, optimizer, epoch, checkpoint_dir)

    epoch += 1
