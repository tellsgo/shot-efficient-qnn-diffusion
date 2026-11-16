#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
if [[ "$PYTHON" == */* ]]; then
  PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"
fi
target=${1:-quick}
if [ "$#" -gt 0 ]; then shift; fi
case "$target" in
  quick)
    "$PYTHON" scripts/validation/shot_noise_sanity.py
    "$PYTHON" scripts/validation/verify_ddim_covariance.py
    "$PYTHON" scripts/validation/check_recorded.py
    ;;
  bounds)
    mkdir -p scripts/results/final_gmm_diagW_pi/diagW
    cp -Rn reference_results/scripts/results/final_gmm_diagW_pi/diagW/bounds scripts/results/final_gmm_diagW_pi/diagW/
    "$PYTHON" scripts/plotting/make_fig_bounds_singlecol.py
    ;;
  gmm) (cd scripts && "$PYTHON" run_T1000_mainline.py "$@") ;;
  image) (cd scripts && "$PYTHON" run_image_paper.py "$@") ;;
  8q) (cd scripts && "$PYTHON" run_8qubit_mainline.py "$@") ;;
  image8q) (cd scripts && "$PYTHON" run_image_paper_8q.py "$@") ;;
  *) echo "Usage: bash run_all.sh {quick|bounds|gmm|image|8q|image8q} [arguments...]" >&2; exit 2 ;;
esac
