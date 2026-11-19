# Survey scope and three design directions

At the first level, distinguish how a video is generated: full-sequence diffusion/flow matching repeatedly updates an entire video latent; autoregressive diffusion combines causal ordering with iterative denoising; direct autoregression predicts discrete tokens conditioned on earlier tokens. This repository evaluates the first group. It does not run an experiment across all three paradigms.

Within full-sequence transformers, the evaluated systems illustrate three compatible design emphases, not mutually exclusive architecture classes:

1. **Compressed visual latents — Wan2.2-TI2V-5B.** A highly compressed video representation shortens the visual sequence and allows a compact dense generator. Fewer tokens reduce attention and sampling work, while reconstruction and fine detail depend on what the representation preserves.
2. **Noise-stage experts — Wan2.2-T2V-A14B.** Separate high- and low-noise experts allocate model capacity to different denoising stages. The tested workflow switches after ten of twenty steps. Only one expert is used at a time; total checkpoint size and memory transfers still matter on one GPU. The additional cost and output quality must be measured together.
3. **Joint audio–video generation — MiniMax-H3.** A shared dense transformer generates the two modalities together. Video-only metrics leave part of its output unevaluated, so the repository also includes an audio–video correspondence experiment. Joint sampling and audio decoding are part of the measured generation interval. Subtracting audio-decoder time would not reconstruct the cost of a video-only transformer.

Wan5B and H3 are not contrasting single-stream versus dual-stream representatives. A14B's experts are selected by noise stage; this should not be described as generic per-token sparse routing. Representation, scale, training, guidance and quantization differ together, so comparisons between these configurations do not isolate the causal effect of one design choice.

The study fixes physical hardware and uses the same formal prompt text, seed policy, encoding and evaluation implementations. It does not force identical frame counts or fps, or pretend that all FP8-labelled files use the same encoder precision. Native outputs are the primary comparison; common presentation is a separate sensitivity check.
