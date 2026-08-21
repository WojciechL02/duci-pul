#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

nruns=5
results_dir=./results/cvpr/all
data_dir=./data/bigvar
target=var_36


for series in deg512; do
  for run_type in real synth ae_synth; do
    for ss_len in 100 500 1000 2000; do
      for method in tice threshold; do
        for p in $(seq 0.1 0.1 1.0); do
          python main.py target=$target prob=$p run_type=$run_type n_runs=$nruns method=$method data_dir=$data_dir/$series results_dir=$results_dir ss_len=$ss_len &
        done
        wait
      done
    done
  done
done
# method.pul_args.pretraining_epochs=250 method.pul_args.pretraining_lr=1e-4


# for target in rar_xxl rar_xl; do
#   for run_type in real synth ae_synth; do
#     for ss_len in 2000; do
#       for method in threshold; do
#         for p in $(seq 0.1 0.1 1.0); do
#           python main.py target=$target prob=$p run_type=$run_type n_runs=$nruns method=$method data_dir=$data_dir results_dir=$results_dir ss_len=$ss_len &
#         done
#         wait
#       done
#     done
#   done
# done

# for target in dit_rf; do
#   for run_type in real synth ae_synth; do
#     for ss_len in 2000; do
#       for method in lbe_dit; do
#         for p in $(seq 0.05 0.05 0.5); do
#           python main.py target=$target prob=$p run_type=$run_type n_runs=$nruns method=$method data_dir=$data_dir results_dir=$results_dir ss_len=$ss_len &
#         done
#         wait
#       done
#     done
#   done
# done
