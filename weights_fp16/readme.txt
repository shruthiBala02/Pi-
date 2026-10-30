PI RUN: WEIGHT FORMATS WITH FP16 VALUES AND TRACKS
==================================================

WHAT THIS RUN DOES
------------------
For each of the 5 paper equations (Burgers, Kovasznay, Beam, Black-Scholes, Reaction rho=5) it runs 5 rows:

  row             method     weights   value   tracks
  baseline_FP32   autograd   FP32      FP32    FP32     (the unrounded FP32 network)
  jets_INT8       jets       INT8      FP16    FP16
  jets_INT32      jets       INT32     FP16    FP16
  jets_INT16      jets       INT16     FP16    FP16
  jets_FP16       jets       FP16      FP16    FP16

The networks are the ones exp25 already made (DA-AdaRound weights). Nothing is trained or rounded on the Pi.
Each row computes u and every derivative the equation needs on the WHOLE grid in ONE inference (no batch sizes).

For every row it measures:
  - runtime: how long one full-grid inference takes (median of 30 timed passes)
  - memory: the model in RAM + the peak memory one full-grid inference adds
  - the error of each solution output against the reference, and how far it is from the FP32 baseline
And the table gives the % change of runtime and memory against the baseline.


WHAT YOU NEED
-------------
1. Raspberry Pi 5 with Raspberry Pi OS 64-bit.
2. The official 27 W USB-C power supply. (The last run showed "under-voltage" flags. A weak supply can slow the CPU.)
3. An active cooler (fan). (The last run reached 80 C on Beam.)
4. Python 3.11.
5. This repo (Pi-) on the Pi. Its folder ablations/ holds the 5 equation folders (the networks from the last
   Pi run, exp25): settings.json, rows.json, data.npz and the .pt row files. This run reads the networks from there.


STEP 1: GET THE CODE
--------------------
On the Pi, in the Pi- repo:

  git pull

You now have, side by side:

  Pi-/
    ablations/        (the networks, with the 5 equation folders)
    weights_fp16/     (this folder)

If ablations is somewhere else, change "networks_dir" in config.yaml to its path.


STEP 2: INSTALL (once)
----------------------
  cd weights_fp16
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt

(You can also reuse the venv of ablations: it has the same packages. Just "source" it instead.)


STEP 3: PREPARE THE PI (before every run)
-----------------------------------------
1. Reboot the Pi:   sudo reboot
   (This clears the old throttle flags, so the record starts clean.)
2. Do not use the Pi for anything else during the run. Close the browser and other programs.
   Best: use it over SSH, with no desktop open.
3. Check the power and heat flags:   vcgencmd get_throttled
   It should say throttled=0x0. If not, check the power supply and the cooler first.


STEP 4: RUN THE PROBE
---------------------
  source venv/bin/activate
  python probe.py

It checks 3 things and writes probe_result.json:
  fp16 matmul      the FP16 matrix multiply works and is accurate (used by the jets_FP16 row)
  memory reading   the memory measurement works on this Pi
  pi readings      temperature, clock and throttle flags can be read

At the end it prints each row and its "matmul path". Expected:
  baseline_FP32 -> fp32
  jets_INT8 / jets_INT32 / jets_INT16 -> decode_per_layer
  jets_FP16 -> fp16   (or decode_per_layer if the fp16 test failed)

If "memory reading" or "pi readings" says FAIL, the run will not start. Fix that first.
Run the probe again if you change torch or the Pi.


STEP 5: RUN EVERYTHING
----------------------
Use tmux (or screen), so the run keeps going if SSH disconnects:

  tmux new -s pirun
  source venv/bin/activate
  bash run_all.sh

(Detach with Ctrl-b then d. Come back with: tmux attach -t pirun)

What happens:
  - Each row runs in its own fresh Python process, baseline first.
  - Before each row, it waits until the CPU is at most 60 C (at most 5 minutes).
  - Each row prints one line: runtime, memory, max temperature, throttled yes/no.
  - A throttled row is run again once. Rows still throttled are listed at the end.
  - At the end it builds the table: results/final_table.csv

Expected time: about 20 to 30 minutes for all 25 rows with a fan (each row about 30 to 60 seconds).
Without a cooler it can take much longer (up to about 2 hours), because each row first waits for the CPU to cool
to 60 C (at most 5 minutes per row).

If it stops (error, power cut, Ctrl-C): just run "bash run_all.sh" again.
Finished rows are skipped; it carries on from where it stopped.
To run ONE row again: delete its .json file in results/<equation>/ and run "bash run_all.sh".
To run one row by hand:   python run_one.py --equation burgers_own --row jets_INT8


STEP 6: GET THE RESULTS
-----------------------
Copy the whole results/ folder to your Mac, for example:
  scp -r pi@<pi-address>:~/Pi-/weights_fp16/results ~/Desktop/pi_results

The main file is:   results/final_table.csv   (opens in Excel)

You can rebuild the table on the Mac too. It needs only numpy and pyyaml (no torch):
  put make_table.py and config.yaml next to the results/ folder, then:   python make_table.py


THE TABLE (results/final_table.csv)
-----------------------------------
One line per equation and row (25 lines).

  Equation, Method, Weight format, Value format, Track format
  Matmul path                      how the value multiply was done (fp32, fp16, or decode_per_layer = weights
                                   turned into float32 one layer at a time)
  Weights storage (bytes)          the weights in their format, scales included (biases not counted)
  Runtime (ms)                     one full-grid inference, median of 30 timed passes
  Runtime change vs baseline (%)   (row - baseline) / baseline x 100. "+" = slower than the baseline
  Memory (bytes)                   model in RAM (weights + scales + biases) + peak memory of one full-grid inference
  Memory change vs baseline (%)    (row - baseline) / baseline x 100. "-" = less memory than the baseline
  Solution outputs                 u, or u, v, p for Kovasznay, or V for Black-Scholes
  Error vs reference, output N (%) relative L2 error of output N against the equation's reference solution
  Change vs baseline, output N (%) relative L2 distance of output N from the FP32 baseline's output
                                   (how much the format damages the solution)
  Throttled during timing          yes / no

Other files in results/<equation>/ (one set per row):
  <equation>__<row>.json           every number: all 30 timings, the 3 memory readings, all errors (also the
                                   derivatives), temperatures, clocks, throttle flags, Pi details
  <equation>__<row>.npz            the computed outputs and derivatives on the grid
  <equation>__<row>__thermal.csv   temperature and clock every 0.5 s during the timing


WHAT IS DIFFERENT FROM THE LAST PI RUN (the fixes)
--------------------------------------------------
1. There is a real baseline: FP32 autograd on the FP32 network. Every row gets % change of runtime and memory, and
   its distance from the baseline's solution.
2. Memory is the inference's own memory:
     - no power logger (it kept ~13 MB of readings in memory last time; energy is not measured in this run)
     - the warm-up passes run before memory is read, so PyTorch's one-time set-up is not counted
     - big memory blocks are returned to the system when freed, and the peak is reset just before the pass, so the
       reading is only that one pass
     - no batch-size sweep: one full-grid inference
3. 3 threads instead of 4: one core is left for the system, so passes do not get stuck waiting (last time some rows
   jumped to 2x slower).
4. Timing in 5 blocks of 6 passes, 2 s apart (30 passes), instead of one short block. The median of all 30 is used;
   each block's median is in the .json, so a slow block can be seen.
5. Cool-down before each row (at most 60 C), throttled rows run again, and the readme asks for a reboot, the official
   power supply and a cooler.
6. The final table has the baseline row and the % changes in it.


THINGS TO KEEP IN MIND WHEN READING THE RESULTS
-----------------------------------------------
- The Pi has no INT8/INT16/INT32 x FP16 hardware path, so those rows turn the weights into float32 one layer at a time
  and multiply in float32. Their runtime is the cost of this code, not of special hardware.
- FP16 values and tracks are stored as FP16 between layers; the maths inside a layer is float32, except the value
  multiply of the jets_FP16 row (FP16 x FP16, if the probe said ok).
- One network per equation: the numbers describe these networks, not all networks.
- Jets can need MORE memory than autograd when there are few derivatives (e.g. Reaction has only u_t), and much less
  when there are high derivatives (Beam, Kovasznay). Both are real results.


IF SOMETHING GOES WRONG
-----------------------
"probe_result.json is missing"        run: python probe.py
"the probe ran with torch X"          torch changed: run python probe.py again
"... is missing: networks_dir ..."    ablations is not next to this folder: fix networks_dir in config.yaml
"vcgencmd is missing"                 this is not Raspberry Pi OS: the run needs the Pi
A row says throttled_any=True         check the cooler and power supply; the row is run again once automatically
