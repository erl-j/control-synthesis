# %%
# import preprocessing
import matplotlib.pyplot as plt
import numpy as np


def pack_voices(roll, pitch_image, n_target_voices, debug=False):

    n_timesteps = roll.shape[1]
    n_input_voices = roll.shape[2]

    pitch2voice = {pitch: None for pitch in range(n_input_voices)}

    unused_voices = set(range(n_target_voices))
    used_voices = set()

    new_roll = np.zeros([1, n_timesteps, n_target_voices]).astype("int32")
    new_pitch = np.zeros([1, n_timesteps, n_target_voices]).astype("int32")

    for t in range(n_timesteps):
        if t > 0:
            new_pitch[:, t, :] = new_pitch[:, t-1, :]
        for v in range(n_input_voices):
            if roll[:, t, v] > 0:
                if pitch2voice[v] == None:
                    # ONSET
                    if len(unused_voices) > 0:
                        new_voice = unused_voices.pop()
                    else:
                        # steal a voice
                        pitch2voice
                        new_voice = used_voices.pop()

                    used_voices.add(new_voice)
                    pitch2voice[v] = new_voice
                else:
                    new_voice = pitch2voice[v]
                new_roll[:, t, new_voice] = roll[:, t, v]
                new_pitch[:, t, new_voice] = pitch_image[:, t, v]
            else:
                if pitch2voice[v] != None:
                    # OFFSET
                    new_voice = pitch2voice[v]
                    if new_voice in used_voices:
                        # voice has not been stolen
                        used_voices.remove(new_voice)
                        unused_voices.add(new_voice)

                    pitch2voice[v] = None

    return new_roll, new_pitch


def unpack_voices(packed_roll, packed_pitch_image, debug=False):

    n_timesteps = packed_roll.shape[1]
    n_input_voices = packed_roll.shape[2]

    roll = np.zeros([1, n_timesteps, 128]).astype("int32")

    for t in range(n_timesteps):
        for v in range(n_input_voices):
            if packed_roll[:, t, v] > 0:
                roll[:, t, packed_pitch_image[:, t, v]] = packed_roll[:, t, v]

    return roll
