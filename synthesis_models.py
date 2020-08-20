import os

import ddsp.training
import gin
import librosa
import numpy as np
import tensorboard as tb
import tensorflow as tf
from ddsp.colab import colab_utils
from ddsp.colab.colab_utils import play, specplot
from IPython import get_ipython
from matplotlib import pyplot as plt


def get_trained_synthesis_model(restore_path, CLIP_DURATION=4):

    if CLIP_DURATION != 4:
        gin_file = f"./gin/{CLIP_DURATION}s.gin"
    else:
        # Parse the gin config.
        gin_file = os.path.join(restore_path,
                                'operative_config-0.gin')
    gin.parse_config_file(gin_file)

    # Create Neural Networks.

    model = ddsp.training.models.Autoencoder()

    model.restore(restore_path)

    return model

