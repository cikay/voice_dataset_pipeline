# Voice Dataset Pipeline

A pipeline to process paired audio-text data for any language, suitable for fine-tuning TTS and ASR models. It enhances audio quality, segments long audio into short utterances using CTC forced alignment, filters candidate segments, then clusters speakers.

Configure the target language via `align_language` in `configs/config.yml` (e.g. `kmr` for Kurdish Kurmanji, `eng` for English). See [supported language codes](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner).

## Clone

Clone the latest code:

```bash
git clone https://github.com/cikay/azadiya-welat-voice-dataset-pipeline.git
```

To clone a specific tag (e.g. `v2.0.0`):

```bash
git clone --branch v2.0.0 https://github.com/cikay/azadiya-welat-voice-dataset-pipeline.git
```

## Input Data

Place `dataset/raw_data_manifest.json` before running the pipeline. It is a JSONL file where each line describes one audio/text pair:

```json
{"id": "abc123", "title": "...", "audio_file": "dataset/audio/abc123.wav", "text_file": "dataset/text/abc123.txt"}
```

The `audio` and `text` folders must contain files with the **same stem** so that each manifest entry can reference both.

## Setup

`deepfilternet` requires Rust to compile. Install it before running `pipenv install`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

Then install dependencies:

```bash
pip install pipenv
pipenv shell
pipenv install
```

### RunPod (CUDA Image) Note

If you use `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, keep PyTorch aligned with the image stack:

```bash
pipenv run pip install --no-cache-dir --force-reinstall torch==2.8.0 torchaudio==2.8.0 nvidia-cusparselt-cu12==0.7.1
pipenv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Expected output should start with `2.8.0` and CUDA `12.8`.

## Pipeline

The pipeline is configured via `configs/config.yml`. Run it with:

```bash
python -m voice_dataset_pipeline.pipeline --config configs/config.yml --log-file logs/pipeline.log
```

Stages run in the order defined under the `stages` key in `configs/config.yml`.

To resume from a specific stage (skipping all preceding ones):

```bash
python -m voice_dataset_pipeline.pipeline --config configs/config.yml --from segmentation --log-file logs/pipeline.log
```

### EnhancementStage

Applies audio enhancement tools to each file. Currently supports DeepFilterNet for noise suppression.

Enhancement is conditional — each tool has `run_if` conditions (skip clean audio) and `replace_if` conditions (revert if enhancement damages the signal). Quality is measured using DNS-MOS (`bak`, `sig`, `ovr`, `p808`).

Reads from `dataset/raw_data_manifest.json`.

### SegmentationStage

Splits long audio into short utterances using CTC forced alignment:

1. Loads the [MMS-300M forced alignment model](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) (supports 1,130+ languages).
2. Aligns ground truth text to audio — no ASR transcription involved.
3. Maps word-level timestamps back to sentence boundaries.
4. Sub-splits long sentences by `;:` punctuation.
5. Resamples audio and saves candidate segment WAV files and `metadata.jsonl` to `output_dir`.
6. All parent metadata fields (`audio_id`, `title`, etc.) are inherited by every segment.

| Key | Default | Description |
|---|---|---|
| `end_padding` | `0.15` | Silence padding after each segment end (seconds) |
| `sample_rate` | `24000` | Output sample rate for segment WAV files |

### FilterStage

Runs after segmentation. Filters candidate segments by duration, word count, alignment confidence, and optional text-pattern exclusions.

| Key | Default | Description |
|---|---|---|
| `min_duration` | `2.0` | Minimum segment duration (seconds) |
| `max_duration` | `15.0` | Maximum segment duration (seconds) |
| `min_words` | `3` | Minimum words per segment |
| `min_align_score` | `-7.0` | Minimum alignment confidence score |
| `exclude_abbr` | `true` | Exclude segments containing abbreviation-like uppercase terms |
| `exclude_number` | `true` | Exclude segments containing digits |

### SpeakerClusteringStage

Runs after filtering. Assigns a `speaker_id` to every segment by clustering segments by their source audio file:

1. Groups segments by `audio_id`.
2. Extracts ECAPA-TDNN embeddings (SpeechBrain) for all segments in each group.
3. Computes a centroid embedding per audio file.
4. Clusters audio centroids with agglomerative clustering using cosine distance.
5. Assigns one `speaker_id` per cluster.
6. Stamps `speaker_id` onto all segments from that audio file.

This stage assumes each source audio file contains one reader/speaker.

| Key | Default | Description |
|---|---|---|
| `threshold` | `0.70` | Minimum cosine similarity for merging audio files into the same speaker cluster |
| `id_format` | `speaker_{n:03d}` | Format string for speaker IDs |

## Tests

```bash
pip install pytest
python -m pytest tests/
```

To run a specific test file:

```bash
python -m pytest tests/segmentation/test_stage.py -v
```

## Publish Dataset

Reads the speaker clustering manifest, selects explicit columns, checks that the repo doesn't already exist, then pushes to HuggingFace Hub.

```bash
python -m voice_dataset_pipeline.push_dataset \
  --repo your-username/your-dataset-name
```

`MY_HF_TOKEN` must be set in `.env` or as an environment variable.

## Output Structure

```text
dataset/
├── audio/                            ← input: WAV files
├── text/                             ← input: text files (same stem as audio)
├── raw_data_manifest.json            ← built by build_manifest; read by enhancement
├── enhanced_audio/                   ← enhancement: enhanced + symlinked WAV files
├── audio_segments/                   ← segmentation: segment WAV files
├── enhancement_manifest.json
├── segmentation_manifest.json
├── filter_manifest.json
├── speaker_clustering_manifest.json
└── quality_score_manifest.json
```
