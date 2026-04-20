#!/bin/bash

# for mia in combination; do
#     python create_plots.py ../results/diffusion/uvit/last2/$mia --no-std --ss_len 1000 --run_type real synth ae_synth
#     python create_plots.py ../results/diffusion/uvit/last2/$mia --no-std --ss_len 2000 --run_type real synth ae_synth
#     python create_plots.py ../results/diffusion/uvit/last4/$mia --no-std --ss_len 1000 --run_type real synth ae_synth
#     python create_plots.py ../results/diffusion/uvit/last4/$mia --no-std --ss_len 2000 --run_type real synth ae_synth
# done

# python create_plots.py ../results/diffusion/ditrf/b --no-std --ss_len 2000 --run_type real synth ae_synth

python create_plots.py ../results/cvpr/dit_rf_g --no-std --ss_len 2000 --run_type real synth ae_synth
# python create_plots.py ../results/diffusion/ditrf/sd/multiple_loss/ --no-std --ss_len 1000 --run_type ae_synth
# python create_table.py ../results/cvpr/all/ ../results/cvpr/all/tables/ --output_suffix real --ss_len 1000 --run_type real
# python create_table.py ../results/cvpr/all/ ../results/cvpr/all/tables/ --output_suffix synth --ss_len 1000 --run_type synth
# python create_table.py ../results/cvpr/all/ ../results/cvpr/all/tables/ --output_suffix ae_synth --ss_len 1000 --run_type ae_synth

# python create_sslen_table.py ../results/cvpr/all/ ../results/cvpr/all/tables/ --output_suffix ae_synth --run_type ae_synth

# python create_plots.py ../results/cvpr/ablations/num_nonmem/10/ --no-std --ss_len 2000 --run_type real
# python create_plots.py ../results/cvpr/ablations/num_nonmem/50/ --no-std --ss_len 2000 --run_type real
# python create_plots.py ../results/cvpr/ablations/num_nonmem/100/ --no-std --ss_len 2000 --run_type real
# python create_plots.py ../results/cvpr/ablations/num_nonmem/200/ --no-std --ss_len 2000 --run_type real
# python create_plots.py ../results/cvpr/ablations/num_nonmem/500/ --no-std --ss_len 2000 --run_type real

