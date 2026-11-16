# Leveraging Quantum Measurement Randomness for Shot-Efficient Diffusion Models

## Requirements

Python 3.11 and `pip install -r requirements.txt`. Full training was run on a GPU;
CPU training can take hours to days. The unused auxiliary third-party PyTorch
snapshot was excluded because its upstream license could not be identified.

## One-command reproduction

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh quick
```

Quick runs CPU covariance checks and compares recorded results against the
paper; it reports discrepancies and exits nonzero if comparisons fail. It does
not retrain models. Expected CPU runtime is under 10 minutes. `bash run_all.sh
bounds` plots recorded shot-bound results; `gmm`, `image`, and `8q` pass
additional arguments to the relevant experiment scripts. Full commands are in
`RUN_GUIDE.md`.

Download `reference-weights.tar.gz` from the v1.0 release and unpack it at the
repository root: `tar -xzf reference-weights.tar.gz`. It restores the original
`scripts/results/` paths. Load pickle files only from trusted sources. Logged
results are in `reference_results/`; they do not independently reproduce training.
On one CPU rerun of the supplied three-qubit weights, the 20-step output was
rf 0.945, MMD² 0.00523, and 302 shots, versus the paper's 0.941, 0.0049,
and 303. Additional CPU outputs are in `cpu_results/`; they are not used to
silently replace the original recorded reference values.

Large VAE checkpoints and latent arrays are excluded. Regenerate them with
`python scripts/prepare_data.py --dataset mnist` and `python scripts/train_vae.py`.
MNIST download needs network access. Exact cross-device
agreement is not guaranteed. The VAE training script does not fix NumPy's
shuffle seed, so a regenerated VAE need not match the original encoder. The
supplied QNN image weights were trained in the original latent space: pairing
them with a regenerated VAE is a smoke test, not an exact paper reproduction.

## Figure, table and claim map

Recorded paths below are relative to `reference_results/scripts/results/`.
Some plot scripts expect their inputs under `scripts/results/`: copy the needed
recorded CSVs there first. Weight-based stages also require the release asset.

| Paper item | Command / input and limitation |
|---|---|
| Figs. 1–3 circuit, pipeline, encoding | Conceptual artwork; source renderer not found. |
| Fig. 4 finite-shot approximation | `python scripts/validation/shot_noise_sanity.py` checks the contract, not the original artwork. |
| Fig. 5 GMM triptych | `bash run_all.sh gmm --part triptych --modes diagW --dealias --dealias-max 3.14159 --outdir results/final_gmm_diagW_pi`. |
| Fig. 6 GMM denoising | Same command with `--part infer --infers 20` instead of `--part triptych`; `T20/denoise_4.png`. |
| Fig. 7 shots | `python scripts/plotting/make_fig_shots_multistep.py --src reference_results/scripts/results/final_gmm_diagW_pi/diagW/infer` |
| Fig. 8 allocation | `python scripts/plotting/make_fig_pareto_rf.py`; `final_gmm_diagW_pi/diagW/uniform_diagW.csv`. |
| Fig. 9 bounds | `bash run_all.sh bounds`; `final_gmm_diagW_pi/diagW/bounds/`. |
| Fig. 10 eta | `python scripts/plotting/make_fig_eta_merged.py`; `final_gmm_diagW_pi/diagW/eta_diagW.csv`. |
| Fig. 11 / Table I readout | `python scripts/plotting/make_fig11_bigfont.py --part dump3q`, `--part dump8q`, then `--part plot`; weights required. Compare `final_gmm_diagW_pi/*/infer/T20/metrics.csv` and `final_8q/metrics_T20.txt`. Some entries differ. |
| Fig. 12 / Table II encoding | `bash run_all.sh gmm --part encoding --enc-readout diagW --enc-eval-only --enc-hybrid-dealias --dealias-max 3.14159 --enc-seeds 5 --gen-samples 3000 --outdir results/final_gmm_diagW_pi`; logged `final_gmm_diagW_pi/ablation_encoding_diagW/encoding_T20.csv`. |
| Fig. 13 MNIST | `bash run_all.sh image --part gen --source vae --dim 8 --beta 0.01 --readout diagW --layers 20 --rep 20 --eta 1.0 --gen-samples 1500`, repeat with `--dim 16`, and `bash run_all.sh image8q --part gen --beta 0.01 --rep 20 --eta 1.0 --gen-samples 1500`; then `python scripts/plotting/make_merged_img_vertical.py`. Data, VAE and QNN weights required. |
| Table III resources | `python scripts/validation/check_recorded.py`; gate counts follow 2*nq*L and nq*L; shot counts from GMM and image metrics. Per-dimension execution counts not recorded. |
| Abstract and key text: 303, 1.6M, 5300x; image MMD/shot figures | `python scripts/validation/check_recorded.py`; recorded results and arithmetic only. Classifier confidence fractions lack recorded classifier outputs. |

## Third-party data and models

MNIST is downloaded from OpenML (dataset 554), whose metadata labels its license
`Public` without specifying a standard license; consult the source's terms.
The digits dataset is loaded from scikit-learn (BSD-3-Clause); it is a copy of
the test set from the UCI Optical Recognition of Handwritten Digits dataset,
whose UCI distribution is CC BY 4.0. Dataset attribution and citation:
https://doi.org/10.24432/C50P49. JAX and PennyLane are Apache-2.0 licensed;
their dependencies have separate licenses. No pretrained third-party models
or datasets are bundled.

## Citation

arXiv:2610.41721
