# chmod +x main.sh

data_dir="/home/shan/Projects/datasets/3d_data/super_dataset/grasp1_2_psmnet/grasp1_2/exp"

method=super
phase=test

evaluate_super=1
data_cost=1
corr_cost=0
arap_cost=1
rot_cost=1

export DISPLAY=:0.0
python -W ignore main.py --data_dir=${data_dir} --method=${method} --phase=${phase} \
--evaluate_super=${evaluate_super} --data_cost=${data_cost} --corr_cost=${corr_cost} \
--arap_cost=${arap_cost} --rot_cost=${rot_cost}
