# Bridging the Gap Between Target Networks and Functional Regularization

## Convergence disks

https://github.com/AlexPiche/cleanrl-tmlr/blob/master/Convergence_disks.ipynb

## 4 Rooms

```bash
cd toy_task
git clone https://github.com/zafarali/emdp
python3 main_toy.py --batch_size 32 --discount $discount --epsilon $epsilon --target_update_freq $freq --use_target_net 0 --seed $seed --reg_weight $reg_weight --size 11
```

## Atari Results

The results in the paper can be achieved by running the following command

```bash
python cleanrl/fr_dqn_atari_jax.py
```
