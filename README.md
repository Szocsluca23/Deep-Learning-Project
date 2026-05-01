# Flickr Audio Retrieval with Noisy Pairing

This repository implements a visually-grounded speech retrieval task.
Given a spoken caption, the model retrieves the correct image from a gallery.

Core objective:
- train a speech-image retrieval model on Flickr Audio Captions
- simulate audio-image misalignment (label noise) from 0% to 50%
- measure how retrieval quality degrades

## Project idea (short)

The model learns a shared embedding space for:
- audio captions (mel spectrogram -> audio embedding)
- image features (ResNet-50 feature -> image embedding)

At inference time, retrieval is done by cosine similarity in that shared space.

## Repository layout

- `notebooks/`
  - `flickr_audio_retrieval.ipynb`: main notebook (training, evaluation, analysis)
- `data/`
  - `metadata/`: mapping and text files (`wav2capt.txt`, `wav2spk.txt`, etc.)
  - `raw/`: local datasets and image/audio assets (gitignored)
- `artifacts/`
  - cached tensors and precomputed features (gitignored)

## Current retrieval model implementation

Main architecture:
- Audio branch:
  - input: log-mel spectrogram (40 mel bins, padded/truncated to fixed length)
  - encoder: 1D CNN blocks + BatchNorm + ReLU + pooling + dropout
  - projection: linear layer to 256-d embedding
- Image branch:
  - input: pre-extracted ResNet-50 features (2048-d)
  - projector: 2-layer MLP to 256-d embedding
- Similarity and objective:
  - L2-normalized embeddings
  - symmetric InfoNCE (CLIP-style) with learnable temperature

Training setup (baseline):
- optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- scheduler: CosineAnnealingLR
- epochs: 15
- batch size: 64
- split by image identity: 6000 train / 1000 val / 1000 test images

Data and preprocessing:
- WAV loading via `scipy.io.wavfile`
- resampling to 16 kHz
- custom mel filterbank + log-mel computation (no torchaudio dependency)
- cached mel spectrograms in `artifacts/mel_cache.pt`

## Baseline results 

Test set retrieval, after training with different train misalignment rates:

| Train misalignment | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| 0%  | 0.0360 | 0.1346 | 0.2114 |
| 10% | 0.0324 | 0.1130 | 0.1788 |
| 20% | 0.0210 | 0.0814 | 0.1414 |
| 30% | 0.0102 | 0.0518 | 0.0992 |
| 40% | 0.0080 | 0.0312 | 0.0556 |
| 50% | 0.0028 | 0.0158 | 0.0330 |

Observed trend:
- smooth monotonic degradation as misalignment increases
- R@10 drops from 21.14% to 3.30% between 0% and 50% noise

## Next task: Noise-Robust Training (for teammates)

This is the planned extension beyond the baseline.

### Recommended first method: Small-loss sample selection

Idea:
- noisy pairs tend to have larger training loss
- in each batch or epoch, keep only a fraction of low-loss samples

Implementation sketch:
1. Replace scalar batch loss with per-sample loss.
2. Sort losses and keep the smallest `keep_ratio` fraction.
3. Compute optimizer step only on selected samples.
4. Schedule `keep_ratio` from 1.0 down to `1 - noise_rate` (or a tuned floor).
5. Compare robust vs baseline curves on the same plot.

### Alternative methods to try

- Co-teaching:
  - two models trained jointly
  - each model selects small-loss samples for the other model
- Soft target weighting:
  - weight each sample by confidence or by inverse loss
- Mixup on paired embeddings/features:
  - regularization against noisy supervision

### Suggested evaluation for robust training

Report for each misalignment rate:
- baseline R@1 / R@5 / R@10
- robust R@1 / R@5 / R@10
- absolute gain and relative gain, especially at 30-50% noise

Recommended additional metrics:
- median rank and mean rank
- top-50 hit rate
- matched vs unmatched similarity gap

## Dataset download

The raw data is not tracked in this repo (too large). Download it manually:

- **Flickr8k images** (needed to regenerate `image_features.pt`):  
  https://www.kaggle.com/datasets/adityajn105/flickr8k

- **Flickr Audio Captions Corpus** (the 40k WAV files):  
  https://groups.csail.mit.edu/sls/downloads/flickraudio/

Place images under `data/raw/Images/` and audio under `data/raw/flickr_audio/flickr_audio/wavs/`.

## Quick start

1. Create and activate a Python environment.
2. Install dependencies: PyTorch, NumPy, SciPy, Matplotlib, scikit-learn, tqdm.
3. Download datasets (see above) and place them in `data/raw/`.
4. `artifacts/image_features.pt` is already committed. `mel_cache.pt` will be auto-generated on first run.
5. Run `notebooks/flickr_audio_retrieval.ipynb` from top to bottom.

## Notes

- Notebook paths are resolved relative to repository root.
- GPU is used automatically when CUDA is available.
- `data/raw/` and large artifacts are intentionally gitignored.
