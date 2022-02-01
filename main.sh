export DISPLAY=:0.0

# data_dir="/home/shan/Projects/datasets/3d_data/super_dataset/super_exp"
data_dir="/home/shan/Projects/datasets/3d_data/super_dataset/grasp1_2_psmnet/grasp1_2/exp"

method="super"

evaluate_super=1
data_cost=1
depth_cost=0
corr_cost=1
arap_cost=1
rot_cost=1
track_filename="exp1_data_arap_rot.npy"

python -W ignore main.py --data_dir="${data_dir}" --method=${method} \
--evaluate_super=${evaluate_super} --data_cost=${data_cost} --depth_cost=${depth_cost} \
--corr_cost=${corr_cost} --arap_cost=${arap_cost} --rot_cost=${rot_cost} --track_filename=${track_filename}
