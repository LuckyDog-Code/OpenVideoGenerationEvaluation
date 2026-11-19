# Environment provenance

- ComfyUI: `e9afbccc2669677076feb08e52658f129cd8c12b` (clean tracked checkout in the recorded runs).
- VBench: `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`.
- Hugging Face CLIP ViT-B/32: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- ImageBind-Huge checkpoint SHA256: `d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157`.

The generation and audio-evaluation package snapshots come from separate environments. A machine-local `packaging @ file://` line was removed from the generation snapshot. Do not install both snapshots into a single environment or treat them as cross-platform lockfiles. Obtain compatible CUDA PyTorch builds from the PyTorch distribution channel.

The original ImageBind source checkout revision is not pinned in the exported evaluator. To make a new run reproducible, record the selected source revision alongside its outputs. The recorded checkpoint digest should be checked before using the original experiment weights.

The exported scripts retain the original model computations, assertions and measurement boundaries while adding path arguments. CPU tests check their inputs and syntax. They do not certify a fresh GPU run on another software stack.
