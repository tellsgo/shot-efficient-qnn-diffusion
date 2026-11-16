# Experiment reproduction guide

This guide provides the commands needed to reproduce the main experiments and
experimental figures.

## Data preparation

Generated datasets are not stored in the repository. From the repository root,
prepare both MNIST and scikit-learn digits with:

```bash
python scripts/prepare_data.py
```

The command writes `data/mnist28x28.npz` and `data/digits8x8.npz`. Use
`--dataset mnist` or `--dataset digits` to prepare only one of them. See
[`data/README.md`](data/README.md) for the expected arrays and data types.

## Conventions

The commands below use Bash syntax. Unless otherwise noted, run them from the
`scripts/` directory so that experiment outputs are written under
`scripts/results/`:

```bash
cd scripts
```

The experiments were tested in the `quddpm_gpu` Conda environment:

```bash
conda activate quddpm_gpu
```

GPU execution is strongly recommended for the full training runs. A device can
be selected with:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Image experiments require the generated `data/mnist28x28.npz`. The GMM
experiments generate their synthetic data internally.

## Main GMM experiment

The primary configuration uses a 3-qubit QNN, diagonal trainable readout,
de-aliased hybrid encoding with maximum phase pi, and adaptive measurement
shots.

### Train, infer, and generate the main results

```bash
OUT=results/final_gmm_diagW_pi
S=3.14159

for PART in train infer triptych uniform eta; do
  python -u run_T1000_mainline.py \
    --part "$PART" \
    --modes diagW \
    --dealias \
    --dealias-max "$S" \
    --outdir "$OUT"
done
```

The corresponding full-readout encoding ablation is:

```bash
OUT=results/final_gmm_diagW_pi
S=3.14159

python -u run_T1000_mainline.py \
  --part encoding \
  --enc-readout fullW \
  --enc-embeddings amplitude probability hybrid \
  --enc-times on off \
  --enc-hybrid-dealias \
  --dealias-max "$S" \
  --rep 20 \
  --enc-seeds 5 \
  --gen-samples 3000 \
  --outdir "$OUT"

python -u run_T1000_mainline.py \
  --part encoding \
  --enc-readout fullW \
  --enc-eval-only \
  --enc-embeddings amplitude probability hybrid \
  --enc-times on off \
  --enc-hybrid-dealias \
  --dealias-max "$S" \
  --rep 10 \
  --enc-seeds 5 \
  --gen-samples 3000 \
  --outdir "$OUT"
```

The five parts produce the trained weights, reverse-step inference results,
high-shot/adaptive comparison, adaptive/uniform comparison, and DDIM eta
ablation, respectively.

### Encoding and time-embedding ablation

```bash
OUT=results/final_gmm_diagW_pi
S=3.14159

python -u run_T1000_mainline.py \
  --part encoding \
  --enc-readout diagW \
  --enc-embeddings amplitude probability hybrid \
  --enc-times on off \
  --enc-hybrid-dealias \
  --dealias-max "$S" \
  --rep 20 \
  --enc-seeds 5 \
  --gen-samples 3000 \
  --outdir "$OUT"

python -u run_T1000_mainline.py \
  --part encoding \
  --enc-readout diagW \
  --enc-eval-only \
  --enc-embeddings amplitude probability hybrid \
  --enc-times on off \
  --enc-hybrid-dealias \
  --dealias-max "$S" \
  --rep 10 \
  --enc-seeds 5 \
  --gen-samples 3000 \
  --outdir "$OUT"
```

`--enc-hybrid-dealias` applies phase de-aliasing only to hybrid encoding. Do not
replace it with the global `--dealias` option in this ablation.

## GMM comparison experiments

### Full readout baseline

```bash
OUT=results/final_gmm_fullW_2pi
S=6.28319

for PART in train infer triptych uniform eta; do
  python -u run_T1000_mainline.py \
    --part "$PART" \
    --modes fullW \
    --dealias \
    --dealias-max "$S" \
    --outdir "$OUT"
done
```

### Readout ablation at the same phase scale

```bash
OUT=results/final_gmm_diagW_pi
S=3.14159

python -u run_T1000_mainline.py \
  --part train \
  --modes fullW fixed24I \
  --dealias \
  --dealias-max "$S" \
  --outdir "$OUT"

python -u run_T1000_mainline.py \
  --part infer \
  --modes fullW fixed24I \
  --dealias \
  --dealias-max "$S" \
  --outdir "$OUT"

for REP in 20 10; do
  python -u run_T1000_mainline.py \
    --part compare \
    --compare-modes fixed24I diagW fullW \
    --rep "$REP" \
    --dealias \
    --dealias-max "$S" \
    --outdir "$OUT"
done
```

### Measurement-shot bounds

This command reuses the trained mainline weights and does not retrain the QNN.

```bash
python -u run_gmm_bounds.py \
  --part both \
  --outdir results/final_gmm_diagW_pi \
  --mode diagW \
  --dealias-max 3.14159 \
  --rep 20 \
  --gen-samples 3000 \
  --seeds 5
```

Use `--part mmin` or `--part mmax` to run only one side of the ablation.

For the full-readout control, use its matching output directory and phase
scale:

```bash
python -u run_gmm_bounds.py \
  --part both \
  --outdir results/final_gmm_fullW_2pi \
  --mode fullW \
  --dealias-max 6.28319 \
  --rep 20 \
  --gen-samples 3000 \
  --seeds 5
```

### GMM summary statistics

After the mainline and readout-ablation weights have been generated, print the
readout-matrix norms and the high-shot/adaptive GMM summary used in the paper:

```bash
python -u gmm_stats.py
```

This command reads existing weights and reruns evaluation only; it does not
train a model.

## Per-dimension 8-qubit experiment

```bash
OUT=results/final_8q

python -u run_8qubit_mainline.py --part train --outdir "$OUT"
python -u run_8qubit_mainline.py --part gen --rep 20 --outdir "$OUT"
python -u run_8qubit_mainline.py --part perdim --rep 20 --outdir "$OUT"
```

## Image experiments

The image experiments use the same finite-shot sampling mechanism in PCA or
VAE latent spaces.

### Train the VAE models

```bash
python -u train_vae.py
```

### PCA latent experiment

```bash
python -u run_image_paper.py \
  --part train \
  --source pca \
  --dataset mnist \
  --dim 8 \
  --readout diagW \
  --Ttrain 1000 \
  --layers 20 \
  --iters 50000 \
  --samples 12000 \
  --batch 256 \
  --lr 0.02

for ETA in 1.0 0.5 0.3; do
  python -u run_image_paper.py \
    --part gen \
    --source pca \
    --dataset mnist \
    --dim 8 \
    --readout diagW \
    --layers 20 \
    --rep 20 \
    --eta "$ETA" \
    --gen-samples 1500 \
    --grid-n 10
done
```

### VAE latent experiment with a 3-qubit QNN

```bash
python -u run_image_paper.py \
  --part train \
  --source vae \
  --dim 8 \
  --beta 0.01 \
  --readout diagW \
  --Ttrain 1000 \
  --layers 20 \
  --iters 50000 \
  --samples 12000 \
  --batch 256 \
  --lr 0.02

for ETA in 1.0 0.5 0.3; do
  python -u run_image_paper.py \
    --part gen \
    --source vae \
    --dim 8 \
    --beta 0.01 \
    --readout diagW \
    --layers 20 \
    --rep 20 \
    --eta "$ETA" \
    --gen-samples 1500 \
    --grid-n 10
done
```

### VAE latent experiment with an 8-qubit QNN

```bash
python -u run_image_paper_8q.py \
  --part train \
  --beta 0.01 \
  --Ttrain 1000 \
  --layers 12 \
  --iters 80000 \
  --samples 60000 \
  --batch 128 \
  --lr 0.02

for ETA in 1.0 0.5 0.3; do
  python -u run_image_paper_8q.py \
    --part gen \
    --beta 0.01 \
    --rep 20 \
    --eta "$ETA" \
    --gen-samples 1500 \
    --grid-n 10
done
```

### K=15 latent scaling experiment

`train_vae.py` also produces the 16-coordinate VAE needed for this experiment.
Train and evaluate the 4-qubit QNN as follows:

```bash
python -u run_image_paper.py \
  --part train \
  --source vae \
  --dim 16 \
  --beta 0.01 \
  --readout diagW \
  --Ttrain 1000 \
  --layers 20 \
  --iters 50000 \
  --samples 12000 \
  --batch 256 \
  --lr 0.02

python -u run_image_paper.py \
  --part gen \
  --source vae \
  --dim 16 \
  --beta 0.01 \
  --readout diagW \
  --layers 20 \
  --rep 20 \
  --eta 1.0 \
  --gen-samples 1500 \
  --grid-n 10
```

After the 3-qubit, 4-qubit, and 8-qubit VAE-latent configurations have been
trained, generate the final combined image figure with:

```bash
python -u plotting/make_merged_img_vertical.py
```

### Image recognizability evaluation

After completing the PCA, 3-qubit VAE, and 4-qubit VAE image experiments, run:

```bash
python -u recognizability_eta1.py
```

The script reports MNIST round-trip reconstruction accuracy and classifier
confidence/class coverage for the generated PCA-3q, VAE-3q, and VAE-4q images.

## Experimental figures

Run these commands after the corresponding result files have been generated.

```bash
python plotting/make_fig_shots_multistep.py
python plotting/make_fig_pareto_rf.py
python plotting/make_fig_eta_merged.py
python plotting/make_fig_bounds_singlecol.py
```

These commands redraw the main GMM plots from the corresponding generated CSV
files.

The readout comparison figure is rebuilt in three stages:

```bash
python plotting/make_fig11_bigfont.py --part dump3q
python plotting/make_fig11_bigfont.py --part dump8q
python plotting/make_fig11_bigfont.py --part plot
```

Intermediate point clouds are written to `../results/figures/fig11_rebuild/`.
The final figure is written to `../results/figures/fig_gmm_readout.png`.

## Numerical validation

From the `scripts/` directory used throughout this guide, run the standalone
covariance checks with:

```bash
python validation/shot_noise_sanity.py
python validation/verify_ddim_covariance.py
```

Their default outputs are written to `results/validation/` at the repository
root.

## Generated results

The main experiment commands create outputs under:

```text
scripts/results/final_gmm_diagW_pi/
scripts/results/final_gmm_fullW_2pi/
scripts/results/final_8q/
scripts/results/final_image_paper/
```

Training is the expensive stage. Inference, ablations that reuse weights, and
figure generation can be run independently once the relevant `weights.npz` or
CSV files are available.

Final experimental figures are written to `results/figures/` at the repository
root.
