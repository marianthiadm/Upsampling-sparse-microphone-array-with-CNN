# Upsampling Sparse Microphone Arrays with CNNs

This project tackles the problem of **virtual microphone upsampling**: given a sparse 4-channel tetrahedral microphone array, a CNN is trained to predict the full covariance matrix of a dense 32-channel microphone array. The model operates in the time-frequency domain — it takes the covariance matrix computed from the 4-channel STFT as input and outputs the predicted 32-channel covariance matrix, which can then be used for downstream tasks such as beamforming and sound source localisation.

The pipeline consists of:
1. **Data generation** — computing covariance matrices from raw multi-channel audio (`.wav` files).
2. **Training** — training a CNN to map 4-channel to 32-channel covariance matrices.
3. **Inference** — loading a trained model, running predictions on unseen data, and evaluating RMSE loss.
4. **Beamforming & visualisation** — applying delay-and-sum beamforming to the predicted covariance matrices and overlaying the resulting spatial audio maps on video.

---

## Repository Structure

| File | Description |
|---|---|
| `data_generator.py` | PyTorch `Dataset` class (`eigenmic`) that loads 32-channel `.wav` files, computes STFTs, derives both 4-channel (tetrahedron subset) and 32-channel covariance matrices, flattens the upper triangle into a compact representation, and applies normalisation before returning tensors for training or inference. |
| `train.py` | Training script. Instantiates the dataset/dataloader, initialises the model and Adam optimiser, runs the training loop for 100 epochs, periodically evaluates on the test set, plots and saves covariance matrix comparisons, and checkpoints the model every 10 epochs. |
| `inference.py` | Inference/evaluation script. Loads a saved model checkpoint, runs a full forward pass over the test set, computes per-file MSE and per-channel RMSE, saves covariance plot comparisons (4-ch input vs. 32-ch reference vs. 32-ch predicted), and exports CSV loss files and scatter plots. |
| `model_cov_matrix_base.py` | **Baseline model** — a 6-layer stack of standard `Conv2d` layers (16 → 32 → 64 → 128 → 256 → 512 → 1024 channels) with `ChannelLayerNorm` and `Dropout2d` regularisation. |
| `model_cov_matrix_base_1FDC_k3.py` | **Base + 1 FDC model** — identical architecture to the baseline but replaces the first `Conv2d` with a `FreqDependentConv1D` layer, allowing frequency-specific temporal modelling at the input. |
| `model_cov_matrix_2FDC_k3.py` | **2 FDC model** — replaces both the first and last layers with `FreqDependentConv1D`, bracketing the standard `Conv2d` layers with frequency-aware convolutions. This is the model used by `train.py` by default. |
| `model_cov_matrix_fullyFDC.py` | **Fully FDC model** — every layer is a `FreqDependentConv1D`, applying frequency-specific 1-D temporal convolutions throughout the entire network. |
| `model_cov_matrix_expanded.py` | **Expanded baseline** — a wider 7-layer `Conv2d` network (16 → 64 → 256 → 1024 → 2048 → 4096 → 2048 → 1024) with more parameters for higher model capacity. |
| `model_cov_matrix_1FDC_expanded_k3.py` | **Expanded + 1 FDC model** — the expanded architecture but with a `FreqDependentConv1D` at the input layer. |
| `Beamformer_n_video.py` | Loads predicted covariance matrices (`.npy`) from disk, applies delay-and-sum beamforming across a spherical grid of directions (azimuth × elevation), and calls `overlay.py` to produce video clips with the spatial audio map rendered on top. |
| `overlay.py` | Video rendering utility. Generates per-time-frame beamforming heatmap images, assembles them into a video, and uses `ffmpeg` to overlay the heatmap animation on a reference video clip with the corresponding audio track. |

---

## Dataset Layout

The scripts expect the following directory structure relative to the project root:

```
../dataset/
    eigen_dev_train_splits/   # Training .wav files (32-channel, 24 kHz)
    eigen_dev_test_splits/    # Test .wav files (32-channel, 24 kHz)
    Cx_videos/                # Reference video files (.mp4) for beamforming overlay
    Cx_data/                  # Reference audio files (.wav) for beamforming overlay
```

Each `.wav` file must be a 32-channel recording sampled at **24 kHz**. The data generator automatically selects channels `[5, 9, 25, 21]` to simulate a 4-channel tetrahedral sub-array.

For beamforming, a MATLAB transfer function file `transferFunc32.mat` (containing the `H_mic` array of shape `[directions × microphones × frequencies]`) must be present in the project root.

---

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12 (with CUDA support recommended)
- torchaudio
- einops
- numpy
- matplotlib
- pandas
- scipy
- seaborn
- ffmpeg (system binary, required only for `overlay.py` / `Beamformer_n_video.py`)

### Install Python dependencies

```bash
pip install torch torchaudio einops numpy matplotlib pandas scipy seaborn
```

For CUDA-enabled PyTorch, follow the official installation guide at [pytorch.org](https://pytorch.org/get-started/locally/).

> **Note:** `overlay.py` requires a local `ffmpeg` binary at `FFMPEG/bin/ffmpeg.exe` relative to the project root (Windows path). On macOS/Linux, update the `FFMPEG` variable in `overlay.py` to point to your system's `ffmpeg` (e.g. `/usr/bin/ffmpeg`).

---

## Training

1. **Choose a model** — open `train.py` and update the import at the top to your desired model variant:

   ```python
   # Default (2 FDC)
   from model_cov_matrix_2FDC_k3 import cov_upsam

   # Other options:
   # from model_cov_matrix_base import cov_upsam
   # from model_cov_matrix_base_1FDC_k3 import cov_upsam
   # from model_cov_matrix_fullyFDC import cov_upsam
   # from model_cov_matrix_expanded import cov_upsam
   # from model_cov_matrix_1FDC_expanded_k3 import cov_upsam
   ```

2. **Set dataset paths** — edit the `train_path` and `test_path` variables inside `main()` if your dataset is in a different location.

3. **Run training:**

   ```bash
   python train.py
   ```

   - Training runs for **100 epochs** with the Adam optimiser (lr = 1e-4) and MSE loss.
   - The model is evaluated on the test set every 10 epochs.
   - A checkpoint is saved as `best_model_2FDC_k3.pth` after each evaluation.
   - A loss curve plot (`train_vs_test_loss_2FDC_k3.jpeg`) is saved at the end.

4. **Resume from a checkpoint** — set the `pre_trained` variable at the bottom of `train.py`:

   ```python
   pre_trained = 'best_model_2FDC_k3.pth'  # path to your saved checkpoint
   ```

---

## Inference / Testing

1. **Update the model import** in `inference.py` to match the model you trained:

   ```python
   from model_cov_matrix_2FDC_k3 import cov_upsam   # adjust as needed
   ```

2. **Set the checkpoint path** — update the `torch.load(...)` call to point to your saved `.pth` file:

   ```python
   model.load_state_dict(torch.load("best_model_2FDC_k3.pth", map_location=device)["net"])
   ```

3. **Run inference:**

   ```bash
   python inference.py
   ```

   **Outputs generated:**
   - `FDCsqr_plots/` — side-by-side comparison plots of the 4-ch input, 32-ch reference, and 32-ch predicted covariance matrices (log scale).
   - `Covariance_matrices/` — predicted 32-ch covariance matrices saved as `.npy` files.
   - `RMSE_test_loss/rmse_per_column.csv` — per-sample, per-channel RMSE values.
   - `RMSE_test_loss/gtn'pred_per_column.csv` — per-sample predicted vs. ground-truth channel means.
   - `Average Test Loss Groups.jpeg` — scatter plot of average MSE per file group.
   - `RMSE_loss.jpeg` — RMSE loss curve across test samples.

---

## Beamforming & Video Overlay

After running inference (so that `Covariance_matrices/*.npy` files exist):

```bash
python Beamformer_n_video.py
```

This will:
1. Load the predicted covariance matrices.
2. Apply delay-and-sum beamforming across a 141 × 51 azimuth–elevation grid.
3. Overlay the time-varying spatial audio map on the corresponding reference video clip.
4. Save final videos to the `overlays/` folder.

> **Prerequisites:** `transferFunc32.mat` must be present in the project root, and the `Cx_videos/` and `Cx_data/` dataset folders must be populated. `ffmpeg` must be accessible (see Installation).

---

## Model Architecture Summary

All models share the same high-level design: a **deep 1D/2D convolutional network** operating on covariance matrix features in the frequency–time plane `[batch, channels, freq, time]`.

| Model file | Key design | Input channels | Output channels |
|---|---|---|---|
| `model_cov_matrix_base.py` | All Conv2d | 16 | 1024 |
| `model_cov_matrix_base_1FDC_k3.py` | FDC at input only | 16 | 1024 |
| `model_cov_matrix_2FDC_k3.py` | FDC at input & output | 16 | 1024 |
| `model_cov_matrix_fullyFDC.py` | All FDC layers | 16 | 1024 |
| `model_cov_matrix_expanded.py` | All Conv2d, larger width | 16 | 1024 |
| `model_cov_matrix_1FDC_expanded_k3.py` | FDC at input, larger width | 16 | 1024 |

**`FreqDependentConv1D`** — a key building block that applies an independent 1-D temporal convolution for each frequency bin, implemented efficiently as a grouped `Conv2d`. This is in contrast to standard `Conv2d` which shares weights across all frequencies.

**`ChannelLayerNorm`** — a normalisation layer that applies `LayerNorm` across the channel dimension at every time-frequency bin.

The 16 input channels correspond to the flattened upper triangle of the 4-channel complex covariance matrix (real + imaginary parts concatenated). The 1024 output channels correspond to the flattened upper triangle of the 32-channel complex covariance matrix.

---

## Output Conventions

- **Covariance matrices** are stored as 4-D tensors: `[ch, ch, freq, time]`.
- The upper triangle (with diagonal for real, without for imaginary) is extracted and concatenated to form the compact flat representation used as model input/output.
- All audio is processed in **5-second segments** at **24 kHz**, producing STFT frames with an FFT length of 512 (257 frequency bins) and a 50% hop.
- Covariance matrices are temporally averaged to produce **5 snapshots per second**.
