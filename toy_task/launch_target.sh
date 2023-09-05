#!/usr/bin/env bash

for seed in `seq 0 0`; do
    for discount in 0.9999 0.99; do
        for epsilon in 0.95 0.75 0.5 ; do
            #for freq in 10 25 50 100 250 500 1000 2500 5000 ; do
            for freq in 10000 25000; do

                eai job new --restartable --image registry.console.elementai.com/snow.text2questions/dqn_zoo111:latest \
                            --data snow.apiche.toy_fr_dqn:/home/:rw \
                            --workdir /home/ \
                            --cpu 4 \
                            --gpu 1 \
                            --gpu-model-filter "!A100"\
                            --cuda-version 11.0 \
                            --mem 10 \
                            --bid 43\
                            --tag algo="toy_dqn_diff_epsilon8" \
                            -- python3 main_toy.py --batch_size 32 --discount $discount --epsilon $epsilon --target_update_freq $freq --use_target_net 1 --seed $seed  --size 11
        
            done
        done
    done
done
