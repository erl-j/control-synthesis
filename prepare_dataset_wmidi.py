# %%

import pretty_midi
import os
import matplotlib.pyplot as plt
import numpy as np
from revoice import pack_voices, unpack_voices
import librosa
import re
import ddsp
import random
import argparse


parser = argparse.ArgumentParser(description='A tutorial of argparse!')

parser.add_argument('--input_tfr_path')
parser.add_argument('--output_tfr_path')

args = parser.parse_args()

INPUT_PATH = args.input_tfr_path
OUTPUT_PATH = args.output_tfr_path
MIN_NOTE_LENGTH = 15

N_CLIPS = None

TST_PERCENTAGE = 14
F0_FROM_MIDI = True

pretty_midi.pretty_midi.MAX_TICK = 1e10

DATA_SET_PATH = INPUT_PATH

DATA_SET_NAME = DATA_SET_PATH.replace("/", "_")


TRN_PATH = f"{DATA_SET_PATH}"


REVOICE = True

VOICE_LIMIT = 3

N_VOICES = 1

OUTPUT_NAME = DATA_SET_NAME+"_"+str(VOICE_LIMIT)

MIDI_FRAME_RATE = 250

SAMPLE_RATE = 16000

CLIP_LENGTH = 2

CLIP_SAMPLES = SAMPLE_RATE*CLIP_LENGTH
CLIP_FRAMES = MIDI_FRAME_RATE*CLIP_LENGTH

# %%


def get_sample_filepaths(root_dir):
    midi_filepaths = []
    for fp in os.listdir(root_dir):
        if os.path.isdir(root_dir+"/"+fp):
            midi_filepaths.extend(get_sample_filepaths(root_dir+"/"+fp))
        else:
            name, ext = os.path.splitext(fp)
            if ext == ".wav":
                midi_path = root_dir+"/"+name
                midi_filepaths.append(midi_path)
    return midi_filepaths


base_paths = get_sample_filepaths(TRN_PATH)

print(base_paths)


# get number of voices necessary

max_voices = 0
# FIGURE OUT NUMBER OF VOICES NEEDED

filtered_base_paths = []


if REVOICE:
    for fp in base_paths:

        midi_path = fp+".mid"

        pattern = '([A-F]#?[0-8])v(\d{1,2})'

        clip_max_voices = -1

        if os.path.isfile(midi_path):

            midi = pretty_midi.PrettyMIDI(midi_path)

            piano_roll = np.transpose(midi.get_piano_roll(MIDI_FRAME_RATE))

            print(piano_roll.shape)

            if piano_roll.size != 0:

                clip_max_voices = np.max(np.sum(piano_roll > 0, axis=1))
            else:
                clip_max_voices = 0

        elif re.search(pattern, fp):

            clip_max_voices = 1

        if clip_max_voices > 0:

            if clip_max_voices <= VOICE_LIMIT:
                filtered_base_paths.append(fp)
                max_voices = max(
                    [clip_max_voices, max_voices])

            else:
                print(
                    f"TOO MANY VOICES FOUND {clip_max_voices}, limit is {VOICE_LIMIT}")

        # print(f"clip_max_voices are {clip_max_voices}")
        # print(f"max_voices are {max_voices}")

print(filtered_base_paths)

feature_dicts = []

clip_index = 0

if N_CLIPS is not None:
    filtered_base_paths = filtered_base_paths[:N_CLIPS]


for fp in filtered_base_paths:

    midi_path = fp+".mid"
    wav_path = fp+".wav"

    audio, sr = librosa.core.load(wav_path, sr=16000)

    audio_duration = librosa.core.get_duration(audio, SAMPLE_RATE)

    if os.path.isfile(midi_path):

        midi = pretty_midi.PrettyMIDI(midi_path)

        midi_duration = midi.get_end_time()

        piano_roll = np.transpose(midi.get_piano_roll(
            MIDI_FRAME_RATE)).astype("int32")

    elif re.search(pattern, fp):
        pattern = '([A-F]#?[0-8])v(\d{1,2})'
        m = re.search(pattern, fp)

        note_name = m.group(1)

        print(note_name)
        note_number = pretty_midi.note_name_to_number(note_name)

        velocity_number = int(m.group(2))

        midi_duration = min([CLIP_LENGTH, audio_duration])

        piano_roll = np.transpose(
            np.zeros((128, int(audio_duration)*MIDI_FRAME_RATE)))

        piano_roll[:, note_number] = velocity_number*128/16

    n_midi_voices = piano_roll.shape[1]

    pitch_image = np.linspace(0, n_midi_voices, n_midi_voices).astype("int32")[
        None, None, :]

    base_midi_pitch = np.tile(pitch_image, [1, CLIP_FRAMES, 1])

    n_segments = int(midi_duration//CLIP_LENGTH)

    print(audio_duration)
    print(fp)

    for i in range(n_segments):

        print(f"working on clip {clip_index}")

        org_clip_vel = piano_roll[CLIP_FRAMES *
                                  i: CLIP_FRAMES*(i+1)][None, ...]

        if N_VOICES < max_voices:
            max_voices = N_VOICES
        if REVOICE:
            clip_vel, clip_midi_pitch = pack_voices(
                org_clip_vel, base_midi_pitch, max_voices)

        # reconstruct original roll
        org_clip_vel_hat = unpack_voices(clip_vel, clip_midi_pitch)

        # make sure they are the same

        print("pack test result")
        print(np.sum(np.abs(org_clip_vel-org_clip_vel_hat)))
        print(org_clip_vel[:, 100:150, 40:70])

        clip_audio = audio[CLIP_SAMPLES*i:CLIP_SAMPLES*(i+1)][None, ...]

        print(clip_audio.shape)
        clip_ld = ddsp.spectral_ops.compute_loudness(clip_audio)
        clip_crepe_f0, clip_crepe_confidence = ddsp.spectral_ops.compute_f0(
            clip_audio.squeeze(), sample_rate=SAMPLE_RATE, frame_rate=MIDI_FRAME_RATE)

        print("shapes")
        print(clip_ld.shape)
        print(clip_crepe_f0.shape)

        if F0_FROM_MIDI:
            f0_hz = ddsp.core.midi_to_hz(clip_midi_pitch.squeeze())
        else:
            f0_hz = clip_crepe_f0.squeeze()

        feature_dict = {
            "midi_velocity": clip_vel.squeeze(),
            "loudness_db": clip_ld.squeeze(),
            "midi_pitch": clip_midi_pitch.squeeze(),
            "f0_hz": f0_hz,
            "f0_confidence": clip_crepe_confidence.squeeze(),
            "audio": clip_audio.squeeze(),
        }

        feature_dicts.append(feature_dict)
# %%


def write_ds_npy(example_list, fp):

    out_dict = {}

    for feature in example_list[0].keys():
        feature_list = [example[feature] for example in example_list]
        out_dict[feature] = np.stack(feature_list, axis=0).astype("float32")

    np.save(fp, out_dict)


random.shuffle(feature_dicts)


N_TOTAL = len(list(feature_dicts))

N_TST = int(N_TOTAL*TST_PERCENTAGE/100)

tst_samples = feature_dicts[:N_TST]
trn_samples = feature_dicts[N_TST:]


write_ds_npy(tst_samples, OUTPUT_PATH+"/tst")
write_ds_npy(trn_samples, OUTPUT_PATH+"/trn")

# %%
