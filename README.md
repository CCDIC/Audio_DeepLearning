# Audio_DeepLearning

A PyTorch-based audio classification toolkit that supports multiple audio classification models and data processing methods, facilitating the rapid construction and training of audio classification models.

## Features

- Supports various audio feature extraction methods, including MelSpectrogram, Spectrogram, MFCC, Fbank, etc.
- Integrates multiple classic audio classification models such as TDNN, CAMPPlus, EcapaTdnn, Res2Net, etc.
- Provides data augmentation functions, including speed perturbation, volume perturbation, noise perturbation, reverb perturbation, etc.
- Supports automatic mixed-precision training and Pytorch 2.0 compiler acceleration.
- Includes complete data set processing, model training, and evaluation workflows.

## Environment Requirements

```
numpy>=1.19.2
scipy>=1.6.3
librosa>=0.9.1
soundfile>=0.12.1
soundcard>=0.4.2
resampy>=0.2.2
numba>=0.53.0
pydub~=0.25.1
matplotlib>=3.5.2
pillow>=10.0.1
tqdm>=4.66.3
visualdl==2.5.3
pyyaml>=5.4.1
scikit-learn>=1.0.2
torchinfo>=1.7.2
loguru>=0.7.2
yeaudio>=0.0.2
```

## Installation

```bash
git clone https://github.com/CCDIC/Audio_DeepLearning.git
cd Audio_DeepLearning
pip install -r requirements.txt
python setup.py install
```

## Data Preparation

The `create_data.py` script can help generate data lists for training and testing. It supports different types of datasets:

1. **General audio datasets**: Organize audio files into folders by category, then run:
   ```python
   get_data_list(audio_path='dataset/audio', list_path='dataset')
   ```

2. **Language identification datasets**: For specific language datasets, use:
   ```python
   get_language_identification_data_list(audio_path='dataset/language', list_path='dataset/')
   ```

3. **UrbanSound8K dataset**: To process the UrbanSound8K dataset:
   ```python
   create_UrbanSound8K_list(audio_path='dataset/UrbanSound8K/audio',
                            metadata_path='dataset/UrbanSound8K/metadata/UrbanSound8K.csv',
                            list_path='dataset/result8K')
   ```

You can also download a pre-prepared language dataset using the script:
```bash
./tools/download_language_data.sh
```

## Recording Audio

If you need to record your own audio for testing or training, use the `record_audio.py` script:
```bash
python record_audio.py
```
Follow the prompts to enter the recording duration, and the audio file will be saved in the `dataset/save_audio` directory.

## Model Training

The training process is managed by the `trainer.py` class, which supports various configurations through YAML files (e.g., `configs/eres2net.yml`). Key configurations include:

- Model settings (type, number of classes, etc.)
- Optimizer and learning rate scheduler parameters
- Training parameters (epochs, batch size, mixed precision, etc.)

Example training configuration (from `eres2net.yml`):
```yaml
model_conf:
  model: 'ERes2Net'
  model_args:
    num_class: null  # Automatically determined from label list if null

optimizer_conf:
  optimizer: 'Adam'
  optimizer_args:
    lr: 0.001
    weight_decay: 1e-5
  scheduler: 'WarmupCosineSchedulerLR'

train_conf:
  enable_amp: False
  use_compile: False
  max_epoch: 60
  log_interval: 10
```

## Evaluation Metrics

The primary evaluation metric used is accuracy, calculated in `macls/metric/metrics.py`.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
