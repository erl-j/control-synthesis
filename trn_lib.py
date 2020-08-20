from data import MidiDataTFRecordProvider
from preprocessing import MidiPreprocessor

from ddsp.colab import colab_utils
import ddsp
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow.compat.v2 as tf
import tensorboard as tb
from absl import logging
import time
from ddsp.training.preprocessing import DefaultPreprocessor
import os
from load_midi import load_midi
import io
import librosa
from plot_prediction import plot_prediction
import collections


def tr(x): return '%.6f' % (x)


def write_audio_dict_to_summary(audio_dict: dict, summary_writer, step, SAMPLE_RATE):
    with summary_writer.as_default():
        for key, value in audio_dict.items():
            tf.summary.audio(
                key, value, SAMPLE_RATE, step=step, max_outputs=8)


def save_audio_from_dict(audio_dict: dict, save_path: str, SAMPLE_RATE: int, prefix=""):
    for key, value in audio_dict.items():
        librosa.output.write_wav(
            save_path+"/"+prefix+key+".wav", tf.squeeze(value[0, ...]).numpy(), SAMPLE_RATE, norm=False)


def generate_audio_examples(control_model, synthesis_model, batch, is_midi=False):

    example_dict = {}

    if not is_midi:
        real_performance_batch = batch

        real_performance = synthesis_model.call(batch)

        example_dict = {**example_dict,
                        "real_performance_synthesized": real_performance[..., None]}

    target = batch["audio"][..., None]

    naive = naive_audio_example(
        control_model, synthesis_model, batch)

    generated_performance = generated_performance_audio_example(control_model,
                                                                synthesis_model, batch)

    # pre_vibrato_performance = generated_performance_audio_example(control_model,
    #                                                               synthesis_model, batch, vibrato_level=0.002, vibrato_hz=6.0)

    # post_vibrato_performance = generated_performance_audio_example(control_model,
    #                                                                synthesis_model, batch, vibrato_level=0.002, vibrato_hz=6.0, vibrato_before_synthesis_model=False)

    # eighth_resample_performance = generated_performance_audio_example(control_model,
    #                                                                   synthesis_model, batch, resample_ratio=0.125)

    # quarter_resample_performance = generated_performance_audio_example(control_model,
    #                                                                    synthesis_model, batch, resample_ratio=0.25)

    if is_midi:

        naive = tf.math.reduce_sum(
            naive, axis=0, keepdims=True)

        generated_performance = tf.math.reduce_sum(
            generated_performance, axis=0, keepdims=True)

    example_dict = {**example_dict,
                    "naive": naive,
                    "generated_performance": generated_performance,
                    "target": target,
                    # "generated_performance_w_previbrato": pre_vibrato_performance,
                    # "generated_performance_w_postvibrato": post_vibrato_performance,
                    # "eighth_resample": eighth_resample_performance,
                    # "quarter_resample": quarter_resample_performance
                    }

    return example_dict


def naive_audio_example(control_model, synthesis_model,  batch):

    batch = control_model.preprocess(batch)

    naive_synth_inputs = {
        "ld_scaled": batch["midi_velocity_scaled"],
        "f0_scaled": batch["midi_pitch_scaled"],
        "f0_hz": ddsp.core.midi_to_hz(batch["midi_pitch_scaled"]*127.0)
    }

    naive_audio = synthesis_model.decode(naive_synth_inputs)

    return naive_audio[..., None]


def qt_f0_hz(f0_hz):

    f0_midi = ddsp.core.hz_to_midi(f0_hz)

    f0_midi = tf.math.round(f0_midi)

    f0_hz = ddsp.core.midi_to_hz(f0_midi)

    return f0_hz


def generated_performance_audio_example(control_model, synthesis_model, batch, vibrato_level=0.0, vibrato_hz=0.0, resample_ratio=1.0, vibrato_before_synthesis_model=True):

    performance_params = control_model(batch, training=False)

    n_frames = performance_params["predicted_ld_scaled"].shape[1]

    MIDI_FRAME_RATE = 250.0
    vibrato_unit = tf.math.sin(tf.linspace(
        0.0, n_frames/MIDI_FRAME_RATE, n_frames)*vibrato_hz*2.0*3.14)[None, ..., None]

    vibrato = vibrato_unit*vibrato_level

    vibrato_f0_scaled = vibrato + performance_params["predicted_f0_scaled"]

    ld_scaled = performance_params["predicted_ld_scaled"]

    f0_hz = ddsp.core.midi_to_hz(vibrato_f0_scaled*127.0)

    if vibrato_before_synthesis_model:
        f0_scaled = vibrato_f0_scaled
    else:
        f0_scaled = performance_params["predicted_f0_scaled"]

    if resample_ratio < 1.0:

        ld_scaled = ddsp.core.resample(
            ld_scaled, int(ld_scaled.shape[1]*resample_ratio))
        f0_scaled = ddsp.core.resample(
            f0_scaled, int(f0_scaled.shape[1]*resample_ratio))
        f0_hz = ddsp.core.resample(
            f0_hz, int(f0_hz.shape[1]*resample_ratio))

    synth_inputs = {
        "ld_scaled": ld_scaled,
        "f0_scaled": f0_scaled,
        "f0_hz":  f0_hz
    }

    performance_audio = synthesis_model.decode(synth_inputs)

    performance_params = control_model(batch, training=False)

    return performance_audio[..., None]


def save(model, optimizer, epoch, save_dir):
    """Saves model and optimizer to a checkpoint."""
    # Saving weights in checkpoint format because saved_model requires
    # handling variable batch size, which some synths and effects can't.
    start_time = time.time()
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=save_dir, max_to_keep=5)
    step = epoch
    manager.save(checkpoint_number=step)
    logging.info('Saved checkpoint to %s at step %s', save_dir, step)
    logging.info('Saving model took %.1f seconds',
                 time.time() - start_time)


def restore(model, optimizer, epoch, checkpoint_path):
    """Restore model and optimizer from a checkpoint if it exists."""
    logging.info('Restoring from checkpoint...')
    start_time = time.time()

    # Restore from latest checkpoint.
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer)
    latest_checkpoint = ddsp.training.train_util.get_latest_chekpoint(
        checkpoint_path)
    if latest_checkpoint is not None:
        # checkpoint.restore must be within a strategy.scope() so that optimizer
        # slot variables are mirrored.
        checkpoint.restore(latest_checkpoint)
        logging.info('Loaded checkpoint %s', latest_checkpoint)
        logging.info('Loading model took %.1f seconds',
                     time.time() - start_time)
        epoch = int(latest_checkpoint.split("ckpt-")[1])
    else:
        logging.info('No checkpoint, skipping.')
        epoch = 0

    return epoch


def plot_conditionings(out):

    N_FT_FRAMES = 1000

    out = {key: value[:, 0:N_FT_FRAMES] for key, value in out.items(
    ) if isinstance(value, collections.Hashable)}

    plots = []

    for i in range(out["ld_scaled"].shape[0]):

        fig = plot_prediction(out["f0_scaled"][i, :, 0], out["ld_scaled"][i, :, 0], out["predicted_f0_scaled"][i, :, 0],
                              out["predicted_ld_scaled"][i, :, 0], out["midi_pitch"][i, :, 0], out["midi_velocity"][i, :, 0])

        plots.append(figure2tensor(fig))

        plt.clf()

    return plots


def figure2tensor(fig):
    DPI = 100
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=DPI)
    io_buf.seek(0)
    im = tf.image.decode_png(io_buf.getvalue(), channels=4)
    io_buf.close()
    return im
