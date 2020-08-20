from ddsp.training.data import TFRecordProvider, DataProvider
import tensorflow.compat.v2 as tf
import numpy as np
import fnmatch
import os
import glob
from get_counters import get_counters
from preprocessing import MidiPreprocessor
from intonate import intonate


class MidiDataTFRecordProvider(TFRecordProvider):
    @property
    def features_dict(self):
        """Dictionary of features to read from dataset."""
        return {
            'audio':
                tf.io.FixedLenFeature([self._audio_length], dtype=tf.float32),
            'f0_hz':
                tf.io.FixedLenFeature(
                    [self._feature_length], dtype=tf.float32),
            'f0_confidence':
                tf.io.FixedLenFeature(
                    [self._feature_length], dtype=tf.float32),
            'loudness_db':
                tf.io.FixedLenFeature(
                    [self._feature_length], dtype=tf.float32),
            'midi_pitch':
                tf.io.FixedLenFeature(
                    [self._feature_length], dtype=tf.float32),
            'midi_velocity':
                tf.io.FixedLenFeature(
                    [self._feature_length], dtype=tf.float32),
        }


class NpyDataProvider(DataProvider):
    def __init__(self, base_path=None, preprocess=False, intonate=True):
        """TfdsProvider constructor.

        Args:
            name: TFDS dataset name (with optional config and version).
            split: Dataset split to use of the TFDS dataset.
            data_dir: The directory to read TFDS datasets from. Defaults to
            "~/tensorflow_datasets".
        """

        self.intonate = intonate
        self.preprocess = preprocess


        self.base_path = glob.glob(base_path+"*")[0]

        feature_dict = np.load(self.base_path, allow_pickle=True)

    def get_dataset(self, shuffle=True):
        """Read dataset.

        Args:
            shuffle: Whether to shuffle the input files.

        Returns:
            dataset: A tf.data.Dataset that reads from TFDS.
        """

        feature_dict = np.load(self.base_path, allow_pickle=True).item()

        self.dataset = tf.data.Dataset.from_tensor_slices(
            feature_dict)

        # def add_counters(example):
        #     onset_counter, offset_counter = get_counters(
        #         example["midi_velocity"][None, :, None])
        #     example["onset_counter"] = tf.squeeze(onset_counter)
        #     example["offset_counter"] = tf.squeeze(offset_counter)
        #     return example

        if shuffle:
            self.dataset = self.dataset.shuffle(1000)
        return self.dataset
