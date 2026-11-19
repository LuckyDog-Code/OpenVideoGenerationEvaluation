# Open Video Generation: Survey and Evaluation

Reproducible ComfyUI generation and automatic evaluation for **Wan2.2-TI2V-5B**, **Wan2.2-T2V-A14B**, and **MiniMax-H3** on one **NVIDIA L40S**. The repository focuses on three design directions: compressed visual latents, noise-stage experts, and joint audio–video generation. It contains the complete **30-prompt** evaluation set, not a demonstration subset.

## Evaluated configurations

| Model | Generated size | Frames / fps | Sampling | Generator weights |
|---|---|---|---|---|
| Wan2.2-TI2V-5B | 832 × 480 | 121 / 24 | UniPC, 20 steps, CFG 5, shift 8 | FP16 checkpoint cast to FP8 by loader |
| Wan2.2-T2V-A14B | 832 × 480 | 81 / 16 | Euler, 10 + 10 steps, CFG 3.5, shift 5 | Two scaled FP8 experts |
| MiniMax-H3 | 864 × 480 | 124 / 24 | res_multistep, 20 steps, distilled guidance | Pruned scaled FP8 |

These are tested approximately 480p operating points, not official default resolutions. All formal requests reset seed **20261118**. Equal seed integers do not imply equal initial noise across different latent spaces. FP8 describes generator weights; encoders and arithmetic differ. H3 uses the quantized Qwen3-VL NVFP4/AWQ encoder and generates stereo audio alongside video.

## Repository layout

- `configs/`: archived configurations that governed the formal runs.
- `workflows/`: one actual API workflow per model, with the output path made relative. These are API graphs using our measurement nodes, not UI-layout exports.
- `custom_nodes/`: generation timing, stage instrumentation, PyTorch memory peaks, output checks and fixed video encoding.
- `scripts/`: generation, visual evaluation, audio–video evaluation and paired summaries.
- `data/prompts_30.jsonl`: all 30 prompts, full generation text, shorter evaluation caption, six groups and provenance.
- `data/pilots.jsonl`: separate candidate pilot prompts; see the actual runner behavior below.
- `environment/`: observed environment snapshots and version notes.

There are no model weights, generated clips, ratings, logs, papers, reports or experiment result tables here.

## Generation on Linux

Use Python 3.10+ in a CUDA environment and a single visible L40S. The runners intentionally reject another GPU type. Install ComfyUI at the experiment revision:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
git -C ComfyUI checkout e9afbccc2669677076feb08e52658f129cd8c12b
python -m pip install -r ComfyUI/requirements.txt
python -m pip install -r requirements-generation.txt
python scripts/install_nodes.py --comfy-dir /absolute/path/to/ComfyUI
```

Install the appropriate CUDA PyTorch build separately. Obtain the checkpoints specified in `configs/*.json` from the pinned Hugging Face revisions:

- Wan: [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged), revision `ee6f4a40737a995bf5818954cfce6d59443b0f04`.
- H3: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), revision `4b53a35191437de05370da548e59aeabe2a5a09a`.

Keep the exact filenames in the configs. Place generator checkpoints under `ComfyUI/models/diffusion_models/`, encoders under `models/text_encoders/`, and video/audio VAEs under `models/vae/`. Only the current model's weights need to be installed. Check the model repositories' terms separately.

Run one model at a time, using the same interpreter as ComfyUI:

```bash
python scripts/run_wan5b.py --comfy-dir /absolute/path/to/ComfyUI --pilot
python scripts/run_wan5b.py --comfy-dir /absolute/path/to/ComfyUI
# Substitute run_wanA14B.py or run_h3.py for the other systems.
```

`--output-dir`, `--config`, `--prompts`, `--pilots` and `--max-samples` are configurable. Each process starts ComfyUI and runs one warmup. Formal generation requires a successful pilot marker in that output directory and skips existing completed records. The historical H3 runner uses fixed warmup/pilot captions in its source; Wan runners read the pilot file. No server, queue or account configuration is bundled. A process handles up to 4.8 hours of requests, excluding the last in-flight request; rerun to resume.

## What is timed

The primary wall-clock interval covers text encoding, sampling and VAE decoding, including transfers during generation and H3 audio decoding. CUDA is synchronized at the interval boundaries. Initial checkpoint loading, warmup, queue waiting, MP4 encoding and disk writes are excluded. Internal stage wall times are not separately CUDA-synchronized. PyTorch peak allocated and reserved memory are recorded per request; an external `nvidia-smi` sampler observes job-level GPU use. These are different memory measurements.

Native videos are saved with imageio / libx264, `quality=8`, `yuv420p`, and no macroblock resizing. Native frame counts and fps remain unchanged.

## Automatic evaluation

Use a separate evaluation environment. Clone [VBench](https://github.com/Vchitect/VBench) at `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`, install its dependencies and prepare its metric weights. The visual evaluator checks that revision and requires a clean checkout. Install `requirements-evaluation.txt` after choosing compatible CUDA PyTorch/torchvision versions. Observed snapshots document additional package versions, rather than serving as a universal installation lock.

```bash
python scripts/evaluate_visual.py --config configs/wan5b.json \
  --run-dir runs/wan5b_0d09b9dff832 --vbench-dir /path/to/VBench \
  --cache-dir /path/to/cache --view native --allow-downloads
```

CLIP ViT-B/32 evaluates 16 sampled frames against `evaluation_prompt`, without the common generation suffix. The five VBench dimensions are subject consistency, motion smoothness, dynamic degree, imaging quality and aesthetic quality. This is not an official aggregate VBench benchmark. Native evaluation is primary. `--view standardized` samples 80 frames at 16 fps over the first five seconds and letterboxes to 832 × 480, using a verified lossless RGB proxy. It is a presentation sensitivity check, not equalized generation work.

```bash
python scripts/summarize_visual.py \
  --model 5B=runs/wan5b_0d09b9dff832/evaluation/native \
  --model A14B=runs/wanA14B_1ca8ea139990/evaluation/native \
  --model H3=runs/h3_f0b45beb9e6f/evaluation/native \
  --output outputs/native_summary.json
```

The summary uses 4,000 paired prompt bootstrap resamples within the six groups. It does not treat videos or frames as independent prompts and does not estimate generation-seed variance.

For H3, install the dependencies of [ImageBind](https://github.com/facebookresearch/ImageBind), obtain ImageBind-Huge weights, and run in a separate compatible environment:

```bash
python scripts/evaluate_audio_video.py --imagebind-dir /path/to/ImageBind \
  --weights /path/to/imagebind_huge.pth --run-dir runs/h3_f0b45beb9e6f \
  --output-dir outputs/h3_audio_video
```

The script compares each original video–audio pair against the other 29 soundtracks, performs a 10,000-permutation test, retains the full similarity matrix and checks WAV format, duration, clipping and near-silence. Three two-second windows are encoded per clip. This tests scene-level correspondence, not audio quality or event synchronization. `--smoke` evaluates only the first item.

## Prompt provenance and interpretation

The 30 prompts consist of five prompts in each of six groups: appearance/low motion, single-subject action, camera motion, multiple subjects, spatial relations and temporal/physical sequences. Thirteen shorter captions originate from VBench and seventeen were group-authored. The full generation text includes the same deterministic suffix. Selection and source fields are retained; unrelated historical run paths are removed. The exact formal prompt text and prompt IDs are preserved.

The export parameterizes local paths while preserving the original generation and metric algorithms. Static checks can run on a CPU; reproducing GPU execution requires the environment and checkpoints above. The latest export was not rerun on a GPU. Configuration fields such as `pilot_pending` are historical values from the frozen input, not a claim that the original formal run never completed.

## Papers behind the comparison

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).
- [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314).
- [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982).
- [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665).

This repository does not redistribute upstream implementations or checkpoints. Their licenses apply to those dependencies. No additional license grant is assigned to this project's original code yet.
