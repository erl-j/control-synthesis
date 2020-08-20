import matplotlib.pyplot as plt
from ddsp.training.data import DataProvider
import numpy as np
import tensorflow.compat.v2 as tf
import gin


@gin.configurable
class PolyDataProvider(DataProvider):
    def __init__(self, base_path=None):
        """TfdsProvider constructor.

        Args:
            name: TFDS dataset name (with optional config and version).
            split: Dataset split to use of the TFDS dataset.
            data_dir: The directory to read TFDS datasets from. Defaults to
            "~/tensorflow_datasets".
        """

        self.base_path = base_path

    def get_dataset(self, shuffle=True):
        """Read dataset.

        Args:
            shuffle: Whether to shuffle the input files.

        Returns:
            dataset: A tf.data.Dataset that reads from TFDS.
        """

        audio_np = np.load(self.base_path+"_AUDIO.npy")
        ld_np = np.load(self.base_path+"_LD.npy")
        f0_np = np.load(self.base_path+"_F0.npy")

        print(np.mean(audio_np))
        print(np.max(ld_np))
        print(np.mean(f0_np))

        audio = tf.convert_to_tensor(audio_np)
        ld = tf.convert_to_tensor(ld_np)
        f0 = tf.convert_to_tensor(f0_np)

        self.dataset = tf.data.Dataset.from_tensor_slices(
            {"audio": audio, "midi_pitch": f0, "midi_velocity": ld})

        if shuffle:
            self.dataset = self.dataset.shuffle(1000)
        return self.dataset

    def get_one_example_as_batch(self):
        poly_example = next(iter(self.get_dataset().take(1)))

        n_voices = poly_example["midi_pitch"].shape[1]

        poly_example["midi_pitch"] = tf.transpose(poly_example["midi_pitch"])

        poly_example["midi_velocity"] = tf.transpose(
            poly_example["midi_velocity"])

        plot_idx = 0
        plt.subplot(2, 2, 1)
        plt.plot(poly_example["midi_pitch"][plot_idx, ...])

        plt.subplot(2, 2, 2)
        plt.plot(poly_example["midi_velocity"][plot_idx, ...])

        # poly_example["midi_pitch"] = self.prettify_pitch(
        #     poly_example["midi_pitch"], poly_example["midi_velocity"])

        plt.subplot(2, 2, 3)
        plt.plot(poly_example["midi_pitch"][plot_idx, ...])

        plt.subplot(2, 2, 4)
        plt.plot(poly_example["midi_velocity"][plot_idx, ...])

        plt.savefig("post")

        poly_example["audio"] = tf.tile(
            poly_example["audio"][None, ...], [n_voices, 1])

        return poly_example

    def prettify_pitch(self, pitch_tensor, velocity_tensor):

        # make sure that the pitches are correct at the onsets
        # transitions between pitches during offsets
        # realtime analog would be setting pitch to correct value at the onsets if no note preceeds it
        # and then transitioning into the encoder outputted pitch

        n_voices = pitch_tensor.shape[0]
        out_pitch = []
        for v in range(n_voices):

            pitch_vector = pitch_tensor[v, :].numpy().squeeze()
            velocity_vector = velocity_tensor[v, :].numpy().squeeze()

            n_timesteps = pitch_vector.shape[0]

          # first pitch
            t = 0

            while velocity_vector[t] < 0.5:
                t += 1
            start_pitch = pitch_vector[t]
            first_pitch_t = t

            pitch_vector[0:first_pitch_t] = start_pitch

            t = n_timesteps-1

            while velocity_vector[t] < 0.5:
                t -= 1
            end_pitch = pitch_vector[t]
            last_pitch_t = t
            pitch_vector[last_pitch_t:] = end_pitch

            zerots = np.argwhere(pitch_vector < 0.5)

            out_pitch.append(tf.convert_to_tensor(pitch_vector)[None, ...])

        new_pitch_tensor = tf.squeeze(tf.stack(out_pitch, axis=0))

        print(new_pitch_tensor.shape)

        return new_pitch_tensor
