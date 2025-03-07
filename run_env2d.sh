#!/bin/bash

for seed in {0..9}
do
    python main.py --save_model --policy SAC --seed $seed --start_steps 1000 --max_episodes 1001 --expl_noise 0.0 --no_termination_training --max_episode_steps 100 --env Env2D_plus_ten &

    python main.py --save_model --policy SAC --seed $seed --start_steps 1000 --max_episodes 1001 --expl_noise 0.0 --no_termination_training --max_episode_steps 100 --env Env2D_plus_one &
    
    wait  # Ensure both processes finish before moving to the next seed
done
