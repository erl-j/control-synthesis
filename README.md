# Control synthesis

## Usage

### 1-Prepare dataset

put 16k mono audio in a directory like so:

datasets/<dataset_name>/audio_16k

from root directory:

`sh scripts/generate_controlsynthesiss_dataset.sh <dataset_name>`

### 2-Train synthesis model

`sh scripts/train_synthesis_model.sh <dataset_name>`

### 3-Train control model
`sh scripts/train_control_model.sh <dataset_name>`


