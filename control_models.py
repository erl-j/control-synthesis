import ddsp
import tensorflow.compat.v2 as tf
from preprocessing import MidiPreprocessor


class AimuControlModel(tf.keras.Model):

    def __init__(self, n_timesteps, rnn_channels=512, rnn_type='lstm', ch=32, layers_per_stack=1, name=None):
        super().__init__(name=name)

        def stack():
            return ddsp.training.nn.fc_stack(ch, layers_per_stack)

        self.pp = MidiPreprocessor(n_timesteps=n_timesteps)

        self.n_out = 2

        self.pitch_line = tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones())

        self.velocity_line = tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Ones())

        self.input_keys = ("midi_pitch_scaled", "midi_velocity_scaled")
        self.input_stacks = [stack() for k in self.input_keys]
        self.rnn = tf.keras.layers.Bidirectional(
            ddsp.training.nn.rnn(rnn_channels, rnn_type))
        self.out_stack = stack()
        self.dense_out = ddsp.training.nn.dense(self.n_out)

    def call(self, raw_conditioning):

        conditioning = self.pp(raw_conditioning)

        inputs = [conditioning[k] for k in self.input_keys]

        inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]
        x = tf.concat(inputs, axis=(-1))
        x = self.rnn(x)
        x = tf.concat((inputs + [x]), axis=(-1))
        x = self.out_stack(x)
        out = self.dense_out(x)

        out = tf.math.sigmoid(out)

        out_dict = {
            "predicted_ld_scaled": out[..., -1][..., None]+self.velocity_line(conditioning["midi_velocity_scaled"]), "predicted_f0_scaled": out[..., -2][..., None]+self.pitch_line(conditioning["midi_pitch_scaled"])}

        out_dict = {**conditioning, **out_dict}

        return out_dict

    def preprocess(self, features):
        return self.pp(features)
