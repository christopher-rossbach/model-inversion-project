ResNet / CLIP Inversion and Reconstruction
================================================

The project report (methods, quantitative tables, figure discussions) is available at [report/out/report.pdf](report/out/report.pdf).
Not all details are covered in the report (see below).

This repository contains experiments on inverting intermediate feature representations of ResNet / CLIP back to the pixel space using a modular, multi‑objective reconstruction optimizer.

Quick start (core idea):

Given one input image, we optimize either (a) the image pixels directly or (b) a GAN latent so that multiple feature projections (CLIP, ResNet stages, VGG, etc.) of the reconstruction match reference statistics (exact reference embedding or batch average) while regularizing perceptual quality.

Key reconstruction algorithm features (see `inversion/reconstruction.py` and `inversion/run_config.py`):
1. Progressive multi-scale optimization: configurable `scale_steps` (e.g. 32->64->128->224) for coarse-to-fine refinement.
2. Flexible optimization space: direct image tensor or GAN latent (`gan_latent`) with the option to switch to image space mid-run (`optimization_space_changes`).
3. Rich loss suite (individually toggle + scale):
	- Embedding reference losses ("ref") for: CLIP, identity, ResNet18/34/50/101/152, VGG16/19.
	- Embedding batch-average losses ("avg") with optional ReLU strategies: plain removal of last ReLU layers, apply ReLU post-aggregation, or "smart" ReLU that preserves sign consistency.
	- CLIP projection, surrogate CLIP (learned mapping), ResNet projection, ImageNet 1k classification head alignment.
	- Perceptual & structural metrics: LPIPS (alex, vgg, squeeze), SSIM (multi window sizes), total variation (with selectable penalty form), optional TV penalty shaping (linear vs squared ReLU schedule), cosine / MSE / dimension‑normalized MSE distance functions.
4. Gradient normalization: per-sample gradient length normalization to stabilize multi-loss optimization.
5. Data augmentations during optimization (geometric affine jitter) to encourage invariance and robustness.
6. Multi-reconstruction batching: replicate an input to produce several stochastic reconstructions simultaneously.
7. Detailed Weights & Biases logging: per-loss (scaled & unscaled), dynamic learning rate, resolution transitions, intermediate images at a chosen frequency.
8. Configurable learning rate and space schedules (`learning_rate_changes`, `optimization_space_changes`).
9. Optional GAN prior initialization: map ResNet features to a GAN latent via a learned mapper for faster convergence compared to pure Gaussian.
10. Modular network initializers for embedding backbones with optional last-ReLU removal for linear feature behavior.

Where to look for runnable examples:
Minimal runnable experiment scripts live in the SLURM job folders: `slurm/*/script.py` (each folder corresponds to a variant: reference vs average embedding losses, with/without ReLU, distribution shift tests, GAN init, surrogate CLIP, etc.).
Each folder also contains `job.slurm` and `run_jobs.sh` for batch scheduling examples.

Report coverage (what the PDF includes):
- Problem formulation and motivation for feature inversion and the use of foreign embedding space features.
- Description of the loss including the embedding under foreign backbones.
- Experimental comparisons across: reference vs average embedding losses, ReLU vs no-ReLU strategies, backbone variations (ResNet depths, VGG, CLIP).
- Quantitative tables (LPIPS, SSIM, embedding distances).
- Analysis of distribution shift when excluding certain activation nonlinearities.

Not fully covered in the report (but supported in code):
- Surrogate CLIP ("sclip") mapping model to compute a desired CLIP embedding from a ResNet embedding.
- Optimization in GAN latent space.
- Code for GAN training.
- Internal GAN latent mapper architecture and checkpoint management utilities.
- Additional backbone placeholders (identity embedding) used for ablation sanity checks.

High-level usage sketch (pseudo-code):
1. Construct a `RunConfig` specifying which losses to activate and their scales.
2. Build / load a `CLIPSuite` (see utilities in `inversion/clip_utils.py`).
3. Call `run_reconstructions(image_path, config, clip_suite, reconstructions_per_img=N, run_tags=[...])`.
4. Monitor W&B run for intermediate images and loss curves.

Directory pointers:
- Core reconstruction logic: `inversion/reconstruction.py`
- Configuration object: `inversion/run_config.py`
- Supporting utilities & network wrappers: `inversion/utils.py`, `inversion/clip_utils.py`, `inversion/GAN.py`
- SLURM experiment scripts: `slurm/*/script.py`
- Report sources: `report/report.tex` (+ figures & tables in `report/figures/`)
