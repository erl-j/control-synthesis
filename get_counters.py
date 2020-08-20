import tensorflow.compat.v2 as tf


def get_counters(vel):

    n_timesteps = vel.shape[1]
    is_changing = tf.math.abs(tf.pad(vel, ((0, 0), (0, 1), (0, 0))) - tf.pad(vel, ((0, 0),
                                                                                   (1, 0),
                                                                                   (0, 0))))[:, :-1, :] > 0
    # dont forget pitch change!!
    is_activated = vel > 0
    is_onset = tf.math.logical_and(is_activated, is_changing)
    is_offset = tf.math.logical_and(
        tf.math.logical_not(is_activated), is_changing)
    outputs = []
    onset_counters = []
    offset_counters = []
    onset_counter = tf.zeros(vel[:, 0].shape)
    offset_counter = tf.zeros(vel[:, 0].shape)
    EPS = 1
    for t in range(0, n_timesteps):
        onset_counter = tf.where(
            is_onset[:, t], 0.0, onset_counter) + tf.where(is_activated[:, t], 1.0, 0.0)
        offset_counter = tf.where(
            is_offset[:, t], 0.0, offset_counter) + tf.where(is_activated[:, t], 0.0, 1.0)
        onset_counters.append(onset_counter)
        offset_counters.append(offset_counter)

    onset_counters = (tf.stack(onset_counters, axis=1) - 1)/n_timesteps
    offset_counters = tf.stack(offset_counters, axis=1)/n_timesteps
    return onset_counters, offset_counters
