#!/usr/bin/env zsh

for p in $(seq 0.1 0.1 0.9); do
  python3 test_pul_methods.py -data var_24 -nsym 3 -prob $p
done
