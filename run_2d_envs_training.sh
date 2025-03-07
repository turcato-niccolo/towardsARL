envs=("2DGraspEnv" "2DSlideEnv" "2DPlaceEnv")
max_episodes=(300 300 100)

for ((i = 0; i < ${#envs[@]}; i++)); do
  {
  for ((j = 0; j < 10; j++)); do
    for ((k = 0; k < 10; k++)); do
      python main.py --env GeneratedEnv_"${envs[i]}"_${j} --policy SAC --seed ${k} --name logs/"${envs[i]}"/with_terminal_reward/"${envs[i]}"_${j}/SAC/ --expl_noise 0.0 --save_model --max_episodes ${max_episodes[i]}
    done
  done
  }&

  {
  for ((j = 0; j < 10; j++)); do
    for ((k = 0; k < 10; k++)); do
      python main.py --env GeneratedEnv_"${envs[i]}"_NoTermReward_${j} --policy SAC --seed ${k} --name logs/"${envs[i]}"/without_terminal_reward/"${envs[i]}"_NoTermReward_${j}/SAC/ --no_termination_training --expl_noise 0.0 --save_model --max_episodes ${max_episodes[i]}
    done
  done
  }&
done
