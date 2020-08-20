import numpy as np
import pretty_midi
from revoice import pack_voices, unpack_voices
import os

pretty_midi.pretty_midi.MAX_TICK = 1e10


def load_midi_examples(midi_dir, midi_frame_rate, clip_duration, voice_limit, sample_rate):
    out_dict = {}
    for fn in os.listdir(midi_dir):
        batch = load_midi(midi_dir+"/"+fn, midi_frame_rate,
                          clip_duration, voice_limit, sample_rate)
        out_dict[fn] = batch
    return out_dict


def load_midi(midi_path, midi_frame_rate=250, clip_duration=4, n_voices=32, sampling_rate=16000):

    clip_frames = clip_duration*midi_frame_rate

    midi = pretty_midi.PrettyMIDI(midi_path)

    audio = midi.synthesize(fs=16000)[None, ...]

    piano_roll = midi.get_piano_roll(midi_frame_rate)

    padding = clip_frames-piano_roll.shape[1]

    if padding > 0:
        piano_roll = np.pad(piano_roll, ((0, 0), (0, padding)))

    piano_roll = np.transpose(piano_roll).astype("int32")[:clip_frames, :]

    clip_max_voices = np.max(np.sum(piano_roll > 0, axis=1))

    n_midi_voices = piano_roll.shape[1]

    pitch_image = np.linspace(0, n_midi_voices, n_midi_voices).astype("int32")[
        None, None, :]

    base_f0 = np.tile(pitch_image, [1, clip_frames, 1])

    clip_ld, clip_f0 = pack_voices(
        piano_roll[None, ...], base_f0, n_voices)

    # test pack
    new_roll = unpack_voices(clip_ld, clip_f0)

    # push voice dim into batch dim
    clip_ld = np.transpose(clip_ld, [2, 1, 0])
    clip_f0 = np.transpose(clip_f0, [2, 1, 0])

    clip_f0 = remove_0_pitch(clip_f0)

    return {"audio": audio.astype("float32"), "midi_pitch": clip_f0.astype("float32"), "midi_velocity": clip_ld.astype("float32")}


def remove_0_pitch(pitch_vector):
    for v in range(pitch_vector.shape[0]):
        t = 0

        while pitch_vector[v, t] < 0.5:
            t += 1
        start_pitch = pitch_vector[v, t]
        first_pitch_t = t

        pitch_vector[v, 0:first_pitch_t] = start_pitch

        t = pitch_vector.shape[1]-1

        while pitch_vector[v, t] < 0.5:
            t -= 1
        end_pitch = pitch_vector[v, t]
        last_pitch_t = t
        pitch_vector[v, last_pitch_t:] = end_pitch

        for t in range(first_pitch_t, last_pitch_t):
            if pitch_vector[v, t] < 0.5:
                pitch_vector[v, t] == pitch_vector[v, t-1]

    return pitch_vector
