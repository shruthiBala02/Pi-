#!/usr/bin/env bash
# exp18: the 33 equations not yet run on this machine, one after another, each by its own run.sh (5 seeds x arms B, C and the probe's D/E, then summarize.py).
# Run from anywhere: bash run_all_new.sh       (needs probe_result.json: python probe_int8.py first)
# An equation whose <equation>_results/<equation>_summary.csv exists is skipped, so a re-run resumes where it stopped.
# A failed equation is reported and the next one starts; the list of failures is printed at the end.
# Jurj's Ψ-NN has its own run: bash quantized_pinn_rounded_checkpoints/jurej/run.sh
set -uo pipefail
cd "$(dirname "$0")"
[ -f probe_result.json ] || { echo "probe_result.json is missing: run python probe_int8.py first"; exit 1; }
EQUATIONS="poisson_boltzmann_pinnacle poisson_4holes_pinnacle reaction_inverse_abc_deepxde nlse_inverse_beta1
ns_inverse_hayford lorenz_inverse_deepxde burgers_hayford burgers_nu1_pdebench burgers_own_hybrid advection_beta0p1_pdebench
advection_beta0p4_pdebench advection_beta1_pdebench diffreact_nu0p5_rho1_pdebench diffreact_nu0p5_rho10_pdebench
diffreact_nu2_rho10_pdebench wave1d_own wave1d_own_hybrid wave2d_own wave2d_own_hybrid wave_beta5_rathore heat1d_own
heat1d_own_hybrid heat_exchanger_pinnacle heat_multiscale_pinnacle helmholtz_44_acrpinn helmholtz_gradpath
kleingordon_gradpath poisson_5d_pinnacle cavity_re100_pinnacle bl_pe1_nayak bl_pe10_nayak nls_raissi
convection_beta40_rathore"
failed=()
n=0
for eq in $EQUATIONS; do
  n=$((n + 1))
  if [ -f "${eq}_results/${eq}_summary.csv" ]; then echo "== [$n/33] $eq already done"; continue; fi
  echo "== [$n/33] $eq ($(date +%H:%M))"
  bash "quantized_pinn_rounded_checkpoints/$eq/run.sh" || { echo "FAILED: $eq"; failed+=("$eq"); }
done
if [ ${#failed[@]} -eq 0 ]; then echo "all 33 done"; else echo "failed (${#failed[@]}): ${failed[*]}"; exit 1; fi
