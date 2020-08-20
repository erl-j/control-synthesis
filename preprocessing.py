from ddsp.training.preprocessing import Preprocessor, DefaultPreprocessor, at_least_3d
import ddsp
import tensorflow.compat.v2 as tf


class MidiPreprocessor(Preprocessor):
    """Default class that resamples features and adds `f0_hz` key."""

    def __init__(self, n_timesteps=1000):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.df_pp = DefaultPreprocessor(time_steps=n_timesteps)

    def __call__(self, features, training=True):
        super().__call__(features, training)
        return self._default_processing(features)

    def _default_processing(self, features):
        """Always resample to `n_timesteps` and scale 'loudness_db' and 'f0_hz'."""

        # apply preprocesssing (scale loudness and f0, make sure batch dim exists etc..)
        if "loudness_db" in features and "f0_hz" in features:
            features = self.df_pp(features)

        for k in ['midi_velocity', 'midi_pitch']:
            features[k] = at_least_3d(features[k])
            features[k] = ddsp.core.resample(
                features[k], n_timesteps=self.n_timesteps)

        # relu is here to fix an issue in the dataset preparation
        features["midi_velocity_scaled"] = tf.nn.relu(
            features["midi_velocity"]/127.0)
        features["midi_pitch_scaled"] = tf.nn.relu(
            features["midi_pitch"]/127.0)

        return features
