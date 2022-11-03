# Python-SuPer

## Papers.
[[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8977357&casa_token=wgvDhTydU74AAAAA:Z8HMt39ZCuomE0cc8sNhCypmq642tWtcyr5qp99tbTiz3Ojtj1kKeCWYmKUSaNApyb-KgPTPNw&tag=1) Li, Yang, et al. "Super: A surgical perception framework for endoscopic tissue manipulation with surgical robotics." _IEEE Robotics and Automation Letters_ 5.2 (2020): 2294-2301.

[[2]](https://arxiv.org/pdf/2210.16674.pdf)Lin, Shan, et al. "Semantic-SuPer: A Semantic-aware Surgical Perception Framework for Endoscopic Tissue Classification, Reconstruction, and Tracking." _arXiv preprint arXiv:2210.16674_ (2022).
## Create the conda environmemt.
* Creat the conda environment with required packages using the following command:
```
conda env create -f resources/env.yml
```
* If get issues when initialize the environment, then check resources/commands.txt for possible solutions.
## Prepare the data.
* Semantic SuPer data could be downloaded from [here](https://drive.google.com/drive/folders/1cFaTY7_BTcSUA8H-pxc4SQ9svacCIa97?usp=sharing)
* Old SuPer data could be downloaded from [here](https://drive.google.com/drive/folders/13PLGxceow8ekedfq6eoJqRigMfK2zYnK?usp=sharing)
* Pretrain depth estimation & segmentation models could be downloaded from [here](https://drive.google.com/drive/folders/1m6axa92cCHryHF_ZpLH8HAoF6jEcILOf?usp=sharing)

## Run the code
* Run naive SuPer with Monodepth2 depth estimation:
```
mod_id=0
exp_id=0
th_dist=0.1
th_cosine_ang=0.4
mesh_step_size=32
method=super
renderer=pulsar
python main.py --data superv2 --data_dir=${data_dir} \
--method=${method} --mod_id=${mod_id} --exp_id=${exp_id} --renderer ${renderer} --renderer_rad 0.002 \
--num_layers 50 --pretrained_encoder_checkpoint_dir ${pretrained_superv2_depth_checkpoint_dir} \
--depth_model ${depth_model} --pretrained_depth_checkpoint_dir ${pretrained_superv2_depth_checkpoint_dir} --post_process \
--seg_model ${seg_model} --pretrained_seg_checkpoint_dir ${pretrained_superv2_seg_checkpoint_dir} --load_seman_gt \
--sample_dir ${sample_dir} --tracking_gt_file=${trk_gt_file} \
--sf_point_plane --mesh_rot --mesh_face --mesh_step_size ${mesh_step_size} --th_dist ${th_dist} --th_cosine_ang ${th_cosine_ang}
```
_Notes: 1) Model ID (mod_id) and experiment ID (exp_id) are needed as the inputs. They only influence the name of the folder of samples for visualizing the tracking results. 2) data superv2 is for using new SuPer data, and --data superv1 is for using old SuPer data. 3) mesh_step_size is used to control the step size for initializing the grid mesh, see paper [2]. 4) post_process: Use the post processing method to refine the depth input, see paper [2]. 5) --sf_point_plane: ICP loss, --mesh_rot: Rot loss, --mesh_face: Face loss, details can be found in [1, 2]._
* Run Semantic-SuPer:
```
method=seman-super
renderer=pulsar
python main.py --data superv2 --data_dir=${data_dir} \
--method=${method} --mod_id=${mod_id} --exp_id=${exp_id} --renderer ${renderer} --renderer_rad 0.002 \
--num_layers 50 --pretrained_encoder_checkpoint_dir ${pretrained_superv2_depth_checkpoint_dir} \
--depth_model ${depth_model} --pretrained_depth_checkpoint_dir ${pretrained_superv2_depth_checkpoint_dir} --post_process \
--seg_model ${seg_model} --pretrained_seg_checkpoint_dir ${pretrained_superv2_seg_checkpoint_dir} --load_seman_gt \
--sample_dir ${sample_dir} --tracking_gt_file=${trk_gt_file} \
--sf_soft_seman_point_plane --mesh_rot --mesh_face --sf_bn_morph --mesh_step_size ${mesh_step_size} --th_dist ${th_dist} --th_cosine_ang ${th_cosine_ang}
```
_Note: --sf_bn_morph is the semantic-aware morphing loss [2]._
* Plot the tracking results using:
```
python -W ignore evaluate.py --result_folder ${results_folder} --files tracking_rst.npy --files_legends PYSuPer
```

## Tune the model.
The tracking performance is influenced by several parameters:
1. 'depth scale', 'DIVTERM' in utils/config.py/xxxParams
2. Input parameters: loss weights, thresholds (th_xxx)
3. May need to tune 'th' in init_track_pts() for initializing tracked points.
