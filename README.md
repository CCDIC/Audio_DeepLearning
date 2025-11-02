# Audio_DeepLearning

A PyTorch-based audio classification toolkit that supports multiple audio classification models and data processing methods, facilitating the rapid construction and training of audio classification models.

## Features

- Supports various audio feature extraction methods, including MelSpectrogram, Spectrogram, MFCC, Fbank, etc.
- Integrates multiple classic audio classification models such as TDNN, CAMPPlus, EcapaTdnn, Res2Net, PANNs, ResNet-SE, etc.
- Provides data augmentation functions, including speed perturbation, volume perturbation, noise perturbation, reverb perturbation, and spec augmentation.
- Supports automatic mixed-precision training and Pytorch 2.0 compiler acceleration.
- Includes complete dataset processing, model training, evaluation, and feature extraction workflows.
- Allows custom configuration of training parameters through YAML files for different models.

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

The `create_data.py` script helps generate data lists for training and testing, supporting different dataset types:

1. **General audio datasets**: Organize audio files into folders by category, then run:
   ```python
   get_data_list(audio_path='dataset/audio', list_path='dataset')
   ```
   This will generate `train_list.txt`, `test_list.txt`, and `label_list.txt` in the specified `list_path`.

2. **Language identification datasets**: For specific language datasets with predefined label mappings, use:
   ```python
   get_language_identification_data_list(audio_path='dataset/language', list_path='dataset/')
   ```
   Supports 14 language/dialect categories including Standard Mandarin, Southwestern Mandarin, Wu dialect, etc.

3. **UrbanSound8K dataset**: To process the UrbanSound8K dataset with metadata:
   ```python
   create_UrbanSound8K_list(audio_path='dataset/UrbanSound8K/audio',
                            metadata_path='dataset/UrbanSound8K/metadata/UrbanSound8K.csv',
                            list_path='dataset/result8K')
   ```

You can download a pre-prepared language dataset using:
```bash
./tools/download_language_data.sh
```

## Recording Audio

Record custom audio for testing or training with `record_audio.py`:
```bash
python record_audio.py
```
Follow prompts to enter recording duration. Audio files are saved in `dataset/save_audio` with timestamps as filenames.

## Feature Extraction

Pre-extract and save audio features to accelerate training using `extract_features.py`:
```bash
python extract_features.py --configs configs/cam++.yml --save_dir dataset/features --max_duration 100
```
- `--configs`: Path to model configuration file
- `--save_dir`: Directory to save extracted features
- `--max_duration`: Maximum audio duration (in seconds) to prevent memory issues

Extracted features are saved as `.npy` files, with corresponding list files (`train_list_features.txt`, `test_list_features.txt`) generated.

## Model Training

Train models using `train.py` with configuration files in the `configs` directory (supports `eres2net.yml`, `cam++.yml`, `ecapa_tdnn.yml`, `tdnn.yml`, `res2net.yml`, `panns.yml`, `resnet_se.yml`):

```bash
python train.py --configs configs/cam++.yml --save_model_path models/ --log_dir log/
```

Key parameters:
- `--configs`: Path to model configuration file (defines dataset settings, feature extraction method, batch size, etc.)
- `--data_augment_configs`: Path to data augmentation configuration file (default: `configs/augmentation.yml`)
- `--save_model_path`: Directory to save trained models
- `--log_dir`: Directory to save VisualDL logs
- `--resume_model`: Path to resume training from a checkpoint
- `--pretrained_model`: Path to load pretrained weights
- `--overwrites`: Override configuration parameters (e.g., `"train_conf.max_epoch=100"`)

### Configuration Details

Configuration files control training behavior, including:

```yaml
# Dataset parameters
dataset_conf:
  dataset:
    min_duration: 0.4        # Minimum audio duration (seconds)
    max_duration: 3          # Maximum audio duration (seconds)
    sample_rate: 16000       # Audio sampling rate
    use_dB_normalization: True  # Enable volume normalization
    target_dB: -20           # Target volume for normalization
  dataLoader:
    batch_size: 64           # Training batch size
    num_workers: 8           # Number of data loading threads
  eval_conf:
    batch_size: 8            # Evaluation batch size
    max_duration: 20         # Maximum duration for evaluation

# Preprocessing parameters
preprocess_conf:
  use_hf_model: False        # Use HuggingFace models for feature extraction
  feature_method: 'Fbank'    # Feature extraction method (MelSpectrogram, Spectrogram, MFCC, Fbank)
```

## Evaluation Metrics

The primary evaluation metric is accuracy, calculated in `macls/metric/metrics.py`. During training, accuracy and loss are logged for both training and validation sets.

## Project Structure

```
Audio_DeepLearning/
├── create_data.py           # Dataset list generation script
├── extract_features.py      # Audio feature extraction script
├── record_audio.py          # Audio recording script
├── train.py                 # Model training script
├── configs/                 # Configuration files for different models
├── macls/
│   ├── trainer.py           # Training manager class
│   ├── data_utils/
│   │   ├── reader.py        # Dataset loading class
│   │   └── featurizer.py    # Audio feature extraction class
│   ├── metric/
│   │   └── metrics.py       # Evaluation metrics
│   └── utils/
│       └── record.py        # Audio recording utility
└── tools/
    └── download_language_data.sh  # Language dataset download script
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
