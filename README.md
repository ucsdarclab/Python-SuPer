# Semantic-SuPer: A Semantic-aware Surgical Perception Framework for Endoscopic Tissue Identification, Reconstruction, and Tracking

This is the code for "Semantic-SuPer: A Semantic-aware Surgical Perception Framework for Endoscopic Tissue Identification, Reconstruction, and Tracking". | [Paper](https://arxiv.org/pdf/2210.16674.pdf) | [Data](https://drive.google.com/file/d/1u5VxNv65CV-EB0Pj91y1bPEB-0OdvLgB/view?usp=sharing) |

It also has the implementation our previous work: SuPer Framework. | [Website](https://sites.google.com/ucsd.edu/super-framework) | [Data](https://drive.google.com/file/d/1ryOy8dqYh9V2O1u5cXf8spuF3JVT5dS9/view?usp=share_link) |

## Create the conda environmemt.
* Creat the conda environment with required packages using the following command:
```
conda env create -f resources/environment.yaml
```
Note: This command may take 7 hours due to package confliction, will improve soon.

## Run the code
### SuPer
```
python main.py --method super --mod_id 1 --exp_id 1 \
--sample_dir [parent directory of the result folder] \
--data superv1 --data_dir [path to data] --tracking_gt_file rgb/left_pts.npy --start_id 4 \
--load_depth \
--load_seg --del_seg_classes 1 \
--sf_point_plane --mesh_rot --mesh_arap
```
Notes: 
* Model ID ```--mod_id``` and experiment ID ```--exp_id``` only control the name of the folder that includes the tracking results (*i.e.*, result folder).
* ```--data superv1``` is for using SuPer data and ```--data superv2``` is for using Semantic-SuPer data. ```--tracking_gt_file``` is the file of tracking ground truth for evaluation, this file is usually saved under the same directory as the RGB images.
* ```--load_depth``` is called to load the precomputed depths.
* ```--load_seg``` is called to load the precomputed semantic segmentation maps. In SuPer, the segmentation maps are only used to filter out surgical tool (```--del_seg_classes 1```) from surfel tracking.
* ```--sf_point_plane```: Point-to-plane ICP loss, ```--mesh_rot```: Rot loss, ```--mesh_arap```: As-rigid-as-possible loss.
* If ```--use_derived_gradient``` is set, the quaternions & translations will be optimized with Levenberg-Marquardt Algorithm and derived gradient, while if ```--use_derived_gradient``` is not used, the quaternions & translations will be optimized with Adam optimizer and PyTorch autograd.

### Semantic-SuPer:
```
python main.py --method semantic-super --mod_id 1 --exp_id 1 \
--sample_dir [parent directory of the result folder] \
--data superv2 --data_dir ${data_dir} --tracking_gt_file rgb/trial_3_l_pts.npy --num_classes 3 \
--num_layers 50 --pretrained_encoder_checkpoint_dir [Path to pre-trained encoder checkpoint] \
--depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir [Path to pre-trained depth model checkpoint] --use_ssim_conf --post_process \
--load_seg \
--sf_soft_seg_point_plane --render_loss --sf_bn_morph --mesh_rot --mesh_face
```
* Semantic-SuPer shares many parameters with SuPer, here we only introduce those that are different from SuPer.
* ```--num_classes``` is the number of semantic classes, the Semantic-SuPer data has three classes: chicken, beef, and surgical tool.
* We finetune [Monodepth2](https://github.com/nianticlabs/monodepth2) on Semantic-SuPer for depth estimation. Both the encoder and decoder code are from their official implementation. Our Monodepth2 checkpoints can be downloaded from [here](https://drive.google.com/file/d/1_8-TifbIlEegxKCjZpIfMn57sGCt7xHc/view?usp=share_link).
* We use the [Segmentation Models Pytorch (SMP) package](https://github.com/qubvel/segmentation_models.pytorch) for semantic segmentation.
* ```--sf_soft_seg_point_plane```: Semantic-aware point-to-plane ICP loss $\mathcal{L}_{icp}$, ```--render_loss```: Rendering loss $\mathcal{L}_{render}$, ```--sf_bn_morph```: Semantic-aware morphing loss $\mathcal{L}_{morph}$, ```--mesh_face```: Face loss $\mathcal{L}_{face}$.

Ensure that you have the desired segmentation masks ready. You may produce them by modifying and running seg/inference.sh, using the pretrained checkpoints [here](https://drive.google.com/drive/folders/1qzv0KKo_t0VfVQkeNvbeB--4klCDXKiU?usp=sharing).
Alternatively, you may train new checkpoints using the ground truths in the folders above and the seg/train.sh script.

## Tune the model.
The tracking performance is influenced by several parameters:
1. ```--mesh_step_size```: Grid step size to initialize the ED graph. ```--num_neighbors```, ```--num_ED_neighbors```: Number of neighboring surfels and ED nodes.
2. ```--th_dist```, ```--th_cosine_ang```: Thresholds to decide if two surfels could be merged.
3. ```--?_weight```: Weights for the losses.

## Contact
shl102@eng.ucsd.edu, amiao@ucsd.edu
