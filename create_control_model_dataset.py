
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# %%

import tensorflow.compat.v2 as tf
import ddsp
import os
import numpy as np
from matplotlib import pyplot as plt
import ddsp.training
from ddsp.colab import colab_utils
import IPython.display as ipd
import argparse
import librosa
from intonate import intonate

parser = argparse.ArgumentParser(description='A tutorial of argparse!')

parser.add_argument('--input_tfr_path')
parser.add_argument('--output_tfr_path')
parser.add_argument('--librosa_od')
parser.add_argument('--example_secs')

args = parser.parse_args()

INPUT_TFRECORD_PATH = args.input_tfr_path
OUTPUT_TFRECORD_PATH = args.output_tfr_path
LIBROSA_ONSET_DETECTION = args.librosa_od is not None
MIN_NOTE_LENGTH = 5
INTONATION = True

# copy pasted from get midi stats
MAESTRO_VEL_MEAN = 65.02882818182805
MAESTRO_VEL_STD = 19.044096499689005

if LIBROSA_ONSET_DETECTION:
    print("IS USING LIBROSA ONSET DETECTION")
else:
    print("IS NOT USING LIBROSA ONSET DETECTION")

print(ddsp.__file__)

fig = plt.figure(figsize=(16, 6))


data_provider = ddsp.training.data.TFRecordProvider(
    INPUT_TFRECORD_PATH+"/*")
dataset = data_provider.get_batch(batch_size=4, shuffle=False, repeats=1)

concatenated_examples = []


def concatenate_batch_into_sample(batch):
    for feature in batch.keys():
        batch[feature] = tf.reshape(batch[feature], [1, -1])
    return batch


for batch in dataset:
    concatenated_examples.append(concatenate_batch_into_sample(batch))


feature_dict = {}

for feature in concatenated_examples[0].keys():
    feature_list = [example[feature] for example in concatenated_examples]
    feature_dict[feature] = tf.squeeze(tf.stack(
        feature_list, axis=0))

feature_dict["f0_hz"] = feature_dict["f0_hz"].numpy()
if INTONATION:
    for di in range(feature_dict["f0_hz"].shape[0]):
        feature_dict["f0_hz"][di, :] = intonate(
            feature_dict["f0_hz"][di, :])


dataset = tf.data.Dataset.from_tensor_slices(feature_dict)

ex = next(iter(dataset))


assert ex["audio"].shape[0] == 16000*16


# read in dataset

N_TOTAL = len(list(dataset))


N_TST = int(N_TOTAL*20/100)


def tf_diff_axis_0(a):
    return tf.pad(a[1:]-a[:-1], [[0, 1]])


PLOT = False


def get_notes(example):
    is_confident = example["f0_confidence"] > 0.5
    is_confident_int32 = tf.cast(is_confident, "int32")

    positive_confidence_change = tf_diff_axis_0(is_confident_int32) > 0

    negative_confidence_change = tf_diff_axis_0(
        tf.pad(is_confident_int32, [[1, 0]])[:-1]) < 0

    f0_midi = tf.math.round(ddsp.core.hz_to_midi(example["f0_hz"]))

    note_change = tf.math.abs(tf_diff_axis_0(f0_midi)) > 0

    all_onsets = tf.math.logical_or(note_change, positive_confidence_change)

    if LIBROSA_ONSET_DETECTION:

        SAMPLE_RATE = 16000
        FRAME_RATE = 250

        onset_idx = librosa.onset.onset_detect(
            example["audio"].numpy(), sr=SAMPLE_RATE, hop_length=SAMPLE_RATE//FRAME_RATE)

        frame_onsets = np.zeros(note_change.shape)
        for ons in onset_idx:
            if ons < frame_onsets.shape[0]:
                frame_onsets[ons] = 1

        all_onsets = frame_onsets > 0

    if PLOT:

        plt.subplot(8, 1, 1)
        plt.title("f0_confidence")
        plt.plot(example["f0_confidence"])

        plt.subplot(8, 1, 2)
        plt.title("is_onset")
        plt.plot(positive_confidence_change)

        plt.subplot(8, 1, 3)
        plt.title("is_offset")
        plt.plot(negative_confidence_change)

        plt.subplot(8, 1, 4)
        plt.title("combined")
        plt.plot(all_onsets)

        plt.subplot(8, 1, 5)
        plt.title("f0_midi")
        plt.plot(f0_midi)

        plt.subplot(8, 1, 6)
        plt.title("f0_note_change")
        plt.plot(note_change)

        plt.subplot(8, 1, 7)
        plt.title("f0_hz")
        plt.plot(example["f0_hz"])

        plt.subplot(8, 1, 8)
        plt.title("loudness db")
        plt.plot(example["loudness_db"])

        plt.show()

    # remove very

    # find start and stop for each activation

    n_timesteps = example["f0_hz"].shape[0]

    notes = []

    current_note = {"on": None, "off": None}

    for t in range(n_timesteps):
        if all_onsets[t]:
            if current_note["on"] is not None:
                current_note["off"] = t
                notes.append(current_note)
                current_note = {"on": None, "off": None}
            current_note["on"] = t

        elif negative_confidence_change[t]:
            if current_note["on"] is not None:
                current_note["off"] = t
                notes.append(current_note)
                current_note = {"on": None, "off": None}

    # remove short notes

    notes = [note for note in notes if note["off"] -
             note["on"] >= MIN_NOTE_LENGTH]

    # get average loudness of each note
    return notes


ld_means = [[] for i in range(128)]
ld_peaks = [[] for i in range(128)]

print(f"len dataset {len(list(dataset))}")

for example in dataset:
    print(example.keys())

    notes = get_notes(example)

    for note in notes:
        on = note["on"]
        off = note["off"]

        f0_midi = tf.math.round(ddsp.core.hz_to_midi(example["f0_hz"]))

        mean_segment_loudness = tf.math.reduce_mean(
            example["loudness_db"][on:off])

        peak_loudness = tf.math.reduce_max(example["loudness_db"][on:off])

        mean_segment_pitch = tf.math.round(
            tf.math.reduce_mean(f0_midi[on:off]))

        ld_means[int(mean_segment_pitch)].append(float(mean_segment_loudness))

        ld_peaks[int(mean_segment_pitch)].append(float(peak_loudness))


# %%


for l in [ld_means, ld_peaks]:

    ld_submeans = [np.mean(ld) for ld in l]

    ld_std = [np.std(ld) for ld in l]

    plt.title("means")

    plt.plot(ld_submeans)
    plt.show()

    plt.title("std")

    plt.plot(ld_std)
    plt.show()


def flatten(l): return [item for sublist in l for item in sublist]


all_peaks = flatten(ld_peaks)

plt.hist(all_peaks, 128)


peak_mean = np.mean(all_peaks)
peak_std = np.std(all_peaks)


# %%


def play_audio(array, sample_rate):
    ipd.display(ipd.Audio(array.numpy(),
                          rate=sample_rate, autoplay=True))


vel_means = [[] for i in range(128)]

DEBUG = False


def augment_example(example):

    notes = get_notes(example)

    estimated_velocity = np.zeros(example["loudness_db"].shape)
    estimated_midi_note = np.zeros(example["loudness_db"].shape)

    for note_idx, note in enumerate(notes):

        on = note["on"]
        off = note["off"]

        # maintain pitch when notes are offset

        if note_idx == 0:
            pitch_ons = 0
        else:
            pitch_ons = note["on"]

        if note_idx < len(notes)-1:
            pitch_offs = notes[note_idx+1]["on"]
        else:
            pitch_offs = estimated_velocity.shape[0]

        mean_segment_loudness = tf.math.reduce_mean(
            example["loudness_db"][on:off])

        peak_loudness = tf.math.reduce_max(example["loudness_db"][on:off])

        mean_segment_pitch = tf.math.round(tf.math.reduce_mean(
            ddsp.core.hz_to_midi(example["f0_hz"][on:off])))

        norm_loudness = (peak_loudness-peak_mean)/peak_std

        midi_vel = (norm_loudness*MAESTRO_VEL_STD)+MAESTRO_VEL_MEAN

        estimated_velocity[on:off] = midi_vel

        estimated_midi_note[pitch_ons:pitch_offs] = mean_segment_pitch

        # vel_means[int(mean_segment_pitch)].append(float(midi_vel))

    example["midi_pitch"] = estimated_midi_note

    example["midi_velocity"] = estimated_velocity

    if DEBUG:

        pitch_hz = ddsp.core.midi_to_hz(estimated_midi_note)[None, ..., None]

        amplitudes = (estimated_velocity /
                      tf.reduce_max(estimated_velocity))[None, ..., None]

        audio = ddsp.core.harmonic_synthesis(
            pitch_hz, amplitudes, None, None, 64000, 16000)

        play_audio(audio, 16000)

        play_audio(example["audio"], 16000)

    return example


tst_set = dataset.take(N_TST)

trn_set = dataset.skip(N_TST)

tst_samples = [augment_example(ex) for ex in tst_set]

trn_samples = [augment_example(ex) for ex in trn_set]


# %%
FP = OUTPUT_TFRECORD_PATH


if not os.path.exists(FP):
    os.makedirs(FP)


def write_ds_npy(example_list, fp):

    out_dict = {}

    for feature in example_list[0].keys():
        feature_list = [example[feature] for example in example_list]
        out_dict[feature] = np.stack(feature_list, axis=0).astype("float32")

    np.save(fp, out_dict)


def write_ds(example_list, fp):

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    si = 0
    with tf.io.TFRecordWriter(fp) as writer:

        for sample in example_list:
            si += 1
            features = {
                'audio':
                _float_feature(sample["audio"]),
                'f0_hz':
                    _float_feature(sample["f0_hz"]),
                'f0_confidence':
                _float_feature(sample["f0_confidence"]),
                'loudness_db':
                _float_feature(sample["loudness_db"]),
                'midi_velocity':
                _float_feature(sample["midi_velocity"]),
                'midi_pitch':
                _float_feature(sample["midi_pitch"])

            }

            example = tf.train.Example(
                features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())


write_ds_npy(tst_samples, FP+"/tst")

write_ds_npy(trn_samples, FP+"/trn")


# %%


# %%
