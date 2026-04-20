#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

nruns=5
results_dir=./results/cvpr/dit_rf_g
data_dir=./data/diffusion/ditrf/g
# target=dit_rf_b

 
# run_type=real
# ss_len=2000
# for method in lbe_lr; do
#   for p in $(seq 0.1 0.1 1.0); do
#     python main.py target=$target prob=$p run_type=$run_type n_runs=$nruns method=$method data_dir=$data_dir results_dir=$results_dir ss_len=$ss_len &
#   done
#   wait
# done

for target in dit_rf_g; do
  for run_type in real synth ae_synth; do
    for ss_len in 2000; do
      for method in tice alphamax lbe_dit threshold; do
        for p in $(seq 0.1 0.1 1.0); do
          python main.py target=$target prob=$p run_type=$run_type n_runs=$nruns method=$method data_dir=$data_dir results_dir=$results_dir ss_len=$ss_len &
        done
        wait
      done
    done
  done
done

# run_type=synth
# for sufix in last4 last6; do
#   for ss_len in 100 500 1000 2000; do
#     for method in tice alphamax lbe threshold; do
#       for p in $(seq 0.1 0.1 1.0); do
#         python main.py target=$target prob=$p run_type=$run_type n_runs=$nruns method=$method data_dir=$data_dir/var/$sufix results_dir=$results_dir/var/$sufix ss_len=$ss_len &
#       done
#       wait
#     done
#   done
# done

# run_type=synth
# for sufix in last2 last4; do
#   for ss_len in 100 500 1000 2000; do
#     for method in tice alphamax lbe threshold; do
#       for p in $(seq 0.1 0.1 1.0); do
#         python main.py target=$target prob=$p run_type=$run_type n_runs=$nruns method=$method data_dir=$data_dir/infinity/$sufix results_dir=$results_dir/infinity/$sufix ss_len=$ss_len &
#       done
#       wait
#     done
#   done
# done
