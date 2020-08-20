import numpy as np
import ddsp


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def intonate(f0_hz):
    N_MIDI_NOTES = 127
    f0_midi = ddsp.core.hz_to_midi(f0_hz).numpy()
    resolution_per_pitch = 16
    smooth_len = 3
    downsampled = (f0_midi*resolution_per_pitch).astype("int32")

    hist = np.zeros(N_MIDI_NOTES*resolution_per_pitch)

    for t in range(downsampled.shape[0]):
        hist[downsampled[t]] += 1

    hist = smooth(hist, smooth_len)

    micro_pitch_sum = np.zeros(N_MIDI_NOTES*resolution_per_pitch)

    for micro_index in range(resolution_per_pitch):
        for pitch in range(N_MIDI_NOTES):
            micro_pitch_sum[micro_index] += hist[pitch *
                                                 resolution_per_pitch+micro_index]

    # print(micro_pitch_sum)

    micro_pitch_offset = np.argmax(micro_pitch_sum)

    # print(micro_pitch_offset)

    new_f0_midi = f0_midi-micro_pitch_offset*1.0/resolution_per_pitch

    new_f0_hz = ddsp.core.midi_to_hz(new_f0_midi)

    return new_f0_hz
