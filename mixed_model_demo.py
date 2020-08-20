from synthesis_models import get_trained_synthesis_model
from control_models import AimuControlModel
from trn_lib import restore, generate_audio_examples, save_audio_from_dict
from data import NpyDataProvider
import tensorflow.compat.v2 as tf
import os
from load_midi import load_midi_examples


CONTROL_MODELS = ["violin3"]

SYNTHESIS_MODELS = ["flute2"]

DATASET = "flute"

CLIP_DURATION = 16
SAMPLE_RATE = 16000

FEATURE_FRAME_RATE = 250

OUT_PATH = "./mixed_model_demo"


data_filepattern = f"./datasets/{DATASET}/control_dataset/tst"

data_provider = NpyDataProvider(data_filepattern)


def get_synthesis_checkpoint_path(name):
    return f"./datasets/{name}/synthesis_checkpoints"


def get_control_checkpoint_path(name):
    return f"./experiments/{name}_AimuControlModel_0/checkpoints"


control_models = {}

for p in CONTROL_MODELS:
    control_model = AimuControlModel(
        n_timesteps=CLIP_DURATION*FEATURE_FRAME_RATE)
    checkpoint_path = f"./experiments/{p}_AimuControlModel_0/checkpoints"

    optimizer = tf.keras.optimizers.Adam(3e-4)

    epoch = 0
    epoch = restore(control_model, optimizer, epoch,
                    checkpoint_path)

    control_models[p] = control_model

synthesis_models = {}

for s in SYNTHESIS_MODELS:
    checkpoint_path = f"./datasets/{s}/synthesis_checkpoints"
    synthesis_model = get_trained_synthesis_model(
        checkpoint_path, CLIP_DURATION=CLIP_DURATION)

    synthesis_models[s] = synthesis_model


dataset = data_provider.get_batch(batch_size=1, repeats=1)

midi_demo_examples = load_midi_examples(
    midi_dir="./midi", midi_frame_rate=FEATURE_FRAME_RATE, clip_duration=CLIP_DURATION, voice_limit=1, sample_rate=SAMPLE_RATE)


batch = next(iter(dataset))

print(control_models)
print(synthesis_models)

for pname, pmodel in control_models.items():
    for sname, smodel in synthesis_models.items():

        tst_examples_audio = generate_audio_examples(
            pmodel, smodel,  batch)

        full_path = OUT_PATH+"/"+f"p:{pname}_s:{sname}_d:{DATASET}"

        midi_path = OUT_PATH+"/"+f"p:{pname}_s:{sname}_midi"
        for path in [full_path, midi_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        save_audio_from_dict(tst_examples_audio,
                             full_path, SAMPLE_RATE)

        midi_demo_audio = {}

        for midi_name, demo_batch in midi_demo_examples.items():
            demo_audio = generate_audio_examples(
                pmodel, smodel, demo_batch, is_midi=True)
            for k in demo_audio.keys():
                midi_demo_audio[f"{midi_name}_{k}"] = demo_audio[k]

        save_audio_from_dict(midi_demo_audio,
                             midi_path, SAMPLE_RATE)
