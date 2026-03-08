# Load Prediction – Semester Project

This project includes a **complete Conda environment** that installs all required packages for the project (including PyTorch, CUDA, NumPy, Pandas, Matplotlib, Scikit-Learn, etc.).

The environment configuration is stored in the file `environment.yml`.

## Recreate the Environment on Your System

1. Make sure [Conda](https://docs.conda.io/en/latest/) is installed.  
2. Open a terminal and navigate to the project directory.  
3. Create the Conda environment from the `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```

4. Activate the environment:

   ```bash
   conda activate sp_project
   ```

5. Download and Prepare Electricity Data

   ```bash
   # Create folder and download dataset
   mkdir -p data && cd data
   zenodo_get -r 3581895

   # Install 7-Zip and extract
   sudo apt install p7zip-full
   7z x Data_vs01.7z

   # Clean up
   rm Data_vs01.7z
   rm -rf rawData/
   cd ..
   ```

6. Download and prepare Weather Data
   ```bash
   cd data/data/
   wget -O weather_data_house_a_LUZ.csv https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/luz/ogd-smn_luz_t_historical_2010-2019.csv
   cd ../..
   ```

---

## Usage – Training and Evaluating Load Forecasting Models

This project is part of the semester project **"Optimal Utilization of a Local Battery in a PV Setup – With Focus on Load Prediction and Optimal Control"**.  
The load forecasting pipeline is built around a **CNN-LSTM neural network** that predicts future household electricity consumption from historical time-series data and weather features.

---
### ⚠️ A warm, heartfelt apology ⚠️

Dear brave soul who just cloned this repository — welcome.

What you are looking at is the proud result of a semester spent simultaneously learning PyTorch, debugging data pipelines at 2 AM, and questioning every life choice that led to this moment😉 . The code works. Mostly. Under the right conditions. With the right environment. 

You will notice things like hardcoded paths scattered across multiple files, configuration variables that live at the very top of each script and absolutely must be set correctly before running anything, model keys that are slightly cryptic (`hor_32_A_no_HP_24_1` — self-explanatory, obviously), and enough commented-out argument parsers to fill a small novel.

This is not *production-grade* code. It is *it-survived-the-semester* code, which is its own remarkable achievement.

Take it one script at a time, read the comments, check the variable names at the top of each file, and you'll get there. Probably. You've got this. 🎉 

PS. Some chatbots are a real lifesaver.

---

Now let's head to the reall explanation of the code.

All scripts are located in `src/` and should be run from the **project root directory**:


### 1. `src/main.py` – Train a new model

This is the main entry point for **training a new forecasting model from scratch**. It loads the energy data, splits it into train/validation/test sets, builds the CNN-LSTM model, trains it, and saves the resulting weights.

**Key configuration** (edit directly at the top of the file before running):

| Variable | Description | Example |
|---|---|---|
| `USE_HOUSE` | Which house dataset to use | `'A'` or `'E'` |
| `FORECAST_HORIZON_TIMESTEPS` | Number of 15-min steps to predict | `16` (4 h), `24` (6 h), `32` (8 h), `48` (12 h) |
| `INPUT_SEQUENCE_LENGTH_TIMESTEPS` | How many past timesteps are fed as input | `192` (= 48 h) |
| `use_cyclic_encoding` | Whether to use cyclic time features (sin/cos) | `True` |

**Run training:**

Run it in your conda environment!

```bash
python src/main.py
```

The trained model weights are saved to `NN_storage/NN_weights/cnn_lstm_forecaster.pth`. Training progress (loss curves) is logged to `runs/` and can be inspected with TensorBoard:

```bash
tensorboard --logdir runs/
```

---

### 2. `src/evaluate_saved_model.py` – Evaluate a saved model

Once a model is trained, use this script to **load it back and inspect its performance** on the test set. It supports several evaluation and plotting modes.

**Key configuration** (edit at the top of the file):

| Variable | Description |
|---|---|
| `USE_HOUSE` | Must match the house used during training |
| `USE_CYCLIC_ENCODING` | Must match the encoding used during training |

**Basic run (plots a few prediction examples):**

```bash
python src/evaluate_saved_model.py \
  --model_path NN_storage/NN_weights/cnn_lstm_forecaster.pth \
  --output_horizon 16
```

**Useful flags:**

| Flag | Description |
|---|---|
| `--plot_full` | Plot the entire test set as one continuous prediction timeline |
| `--plot_full_train` | Same, but for the training set |
| `--plot_full_val` | Same, but for the validation set |
| `--plot_train_val_test` | Plot the raw data split (no model needed) |
| `--predict_at_date 2019-06-01` | Show a prediction starting at a specific date |
| `--n_examples 5` | Number of individual prediction examples to plot |
| `--plot_integrated_difference` | Plot the cumulative energy prediction error |
| `--skip_examples` | Skip individual example plots |

**Example – plot full test set for a 48-step model:**

```bash
python src/evaluate_saved_model.py \
  --model_path house_A/NN_storage_hor_48/NN_storage_8_24er/NN_weights/best_model.pth \
  --output_horizon 48 \
  --plot_full
```

---

### 3. `src/compare_models.py` – Compare multiple models side-by-side

This script loads **multiple pre-trained models** with potentially different forecast horizons and visualises their predictions in a single plot. It is useful for comparing models trained on different houses or with different horizons.

**Key configuration** (edit at the top of the file):

| Variable | Description |
|---|---|
| `HOUSE_TYPE` | Which house data to load for evaluation (`'A'` or `'E'`) |
| `MODELS` | Dictionary of all available model configurations (paths, horizons, colours, …) |

The active models to compare are selected via the `--models` argument. The available model keys are defined in the `MODELS` dict in the script.

**Run example (compare four horizons for House A without heat pump):**

```bash
python src/compare_models.py \
  --models hor_16_A_no_HP_ker5 hor_24_A_no_HP hor_32_A_no_HP_24_1 hor_48_A_no_HP_24_2
```

**Useful flags:**

| Flag | Description |
|---|---|
| `--models <key> [<key> ...]` | Which model(s) from the `MODELS` dict to load and compare |
| `--plot_integrated_difference` | Also plot the cumulative integral error below the main prediction plot |
| `--start_date YYYY-MM-DD` | Restrict the comparison plot to a time window (start) |
| `--end_date YYYY-MM-DD` | Restrict the comparison plot to a time window (end) |

---

### Quick-start summary

```
# 1. Train a new model (edit USE_HOUSE and FORECAST_HORIZON_TIMESTEPS in the file first)
python src/main.py

# 2. Evaluate it on the test set
python src/evaluate_saved_model.py --model_path NN_storage/NN_weights/cnn_lstm_forecaster.pth \
  --output_horizon 16 --plot_full

# 3. Compare several pre-trained models
python src/compare_models.py --models hor_16_A_no_HP_ker5 hor_24_A_no_HP hor_32_A_no_HP_24_1 hor_48_A_no_HP_24_2
```
