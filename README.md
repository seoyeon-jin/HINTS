# HINTS

A model that integrates Human Factor into time series forecasting. Based on TimeMixer, it extracts human decision-making patterns from residual data to improve prediction performance.

## Overview

HINTS (Human-factor INtegrated Time Series) is a framework that improves forecasting performance by learning human decision-making patterns embedded in residual components that cannot be explained by traditional statistical models or machine learning models in time series forecasting.

### Key Features

- **Human Factor Extraction**: Automatically extracts human decision-making patterns from residual data
- **Two-Stage Training Process**: Pre-training of Human Factor extractor → Forecasting model training
- **TimeMixer-based**: Utilizes state-of-the-art time series forecasting architecture
- **Time-Varying FJ Constraint**: Applies time-varying constraint conditions

## Installation

### 1. Environment Setup

```bash
cd src
conda env create --file hints.yml
conda activate timehuman
```

### 2. Additional Package Installation

Install additional packages if needed:

```bash
pip install torch scipy matplotlib
```

## Data Preparation

Place your data in the `dataset/` folder as follows:

```
dataset/
├── PEMS08.npz          # Original time series data
└── resid/
    └── PEMS08.npz      # Residual data (original - predictions)
```

### Data Format

- **Original data**: numpy array with shape `(num_samples, num_features)`
- **Residual data**: residuals obtained by subtracting the base model's predictions from the original data

## Usage

### Basic Execution

Run default experiments on the PEMS08 dataset:

```bash
cd src
sh scripts/pems08.sh
```

### Custom Execution

Run with custom parameter settings:

```bash
cd src
python run.py \
  --task_name 'HumanFactor' \
  --is_training 1 \
  --root_path ../dataset/ \
  --data_path PEMS08.npz \
  --model_id test \
  --model TimeMixer \
  --data human_pems \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --d_model 64 \
  --n_heads 4 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 32 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --human_learning_rate 0.001 \
  --human_train_epochs 20 \
  --human_dim 128 \
  --human_gamma 0.5 \
  --train_epochs 10
```

### Testing Only

To test a trained model:

```bash
python run.py \
  --task_name 'HumanFactor' \
  --is_training 0 \
  --model_id <your_model_id> \
  --model TimeMixer \
  --data human_pems \
  # ... (other parameters should be the same as during training)
```

## Key Parameters

### Time Series Forecasting Parameters

- `--seq_len`: Input sequence length (default: 96)
- `--pred_len`: Prediction sequence length (default: 24)
- `--label_len`: Start token length for decoder (default: 48)
- `--features`: Forecasting type (M: multivariate→multivariate, S: univariate→univariate, MS: multivariate→univariate)

### Model Architecture Parameters

- `--d_model`: Model dimension (default: 16)
- `--n_heads`: Number of attention heads (default: 4)
- `--e_layers`: Number of encoder layers (default: 2)
- `--d_layers`: Number of decoder layers (default: 1)
- `--d_ff`: Feed-forward network dimension (default: 32)
- `--dropout`: Dropout rate (default: 0.1)

### Human Factor Parameters

- `--human_learning_rate`: Human Factor extractor learning rate (default: 0.0001)
- `--human_train_epochs`: Number of pre-training epochs for Human Factor (default: 20)
- `--human_dim`: Human Factor dimension (default: 128)
- `--human_gamma`: Human Factor weight (default: 0.5)
- `--human_coeff`: Human Factor coefficient (default: 0.0)
- `--resid_path`: Residual data path (default: `../dataset/resid/`)

### Training Parameters

- `--learning_rate`: Main model learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 16)
- `--train_epochs`: Number of training epochs (default: 10)
- `--patience`: Early stopping patience (default: 10)

## Project Structure

```
HINTS/
├── dataset/                    # Datasets
│   ├── PEMS08.npz            # Original data
│   └── resid/
│       └── PEMS08.npz        # Residual data
├── src/
│   ├── models/                # Model definitions
│   │   ├── HumanFactor.py    # Human Factor extractor and main model
│   │   └── TimeMixer.py      # TimeMixer architecture
│   ├── data_provider/         # Data loaders
│   │   ├── data_factory.py
│   │   └── data_loader.py
│   ├── exp/                   # Experiment classes
│   │   ├── exp_basic.py      # Base experiment class
│   │   └── exp_human_factor.py  # Human Factor experiment class
│   ├── layers/                # Model layers
│   │   ├── AutoCorrelation.py
│   │   ├── Embed.py
│   │   └── ...
│   ├── utils/                 # Utility functions
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── losses.py         # Loss functions
│   │   └── tools.py          # Tool functions
│   ├── scripts/               # Execution scripts
│   │   └── pems08.sh
│   ├── checkpoints/           # Model checkpoints
│   ├── results/               # Results
│   ├── log/                   # Log files
│   ├── run.py                 # Main execution file
│   └── hints.yml              # Conda environment configuration
└── README.md
```

## Training Process

HINTS follows a two-stage training process:

### Stage 1: Human Factor Extractor Pre-training

1. Use residual data (`resid/`) as input
2. Train Human Factor extractor to extract human decision-making patterns from residuals
3. Learn time-varying constraints through Time-Varying FJ Constraint

### Stage 2: Forecasting Model Training

1. Freeze the weights of the pre-trained Human Factor extractor
2. Train the forecasting model by combining original data with Human Factor
3. The main forecasting model (TimeMixer) performs enhanced predictions utilizing Human Factor

## Result Storage

Training and testing results are saved in the following directories:

- **Checkpoints**: `src/checkpoints/{setting}/`
- **Prediction results**: `src/results/{setting}/`
- **Test results**: `src/test_results/{setting}/`
- **Log files**: `src/log/` or `src/logs/`

## Notes

- GPU usage is recommended (`--use_gpu True`)
- For multi-GPU usage, use `--use_multi_gpu True --devices 0,1` option
- Adjust `enc_in`, `dec_in`, `c_out` parameters appropriately according to your dataset
- Residual data must have the same format and length as the original data
