# Semantic-SuPer: A Semantic-aware Surgical Perception Framework for Endoscopic Tissue Identification, Reconstruction, and Tracking

* This is the code for "Semantic-SuPer: A Semantic-aware Surgical Perception Framework for Endoscopic Tissue Identification, Reconstruction, and Tracking". | [Paper](https://arxiv.org/pdf/2210.16674.pdf) | [Data](https://drive.google.com/file/d/1Pn_E_cH0ES7tfgicSPOQq7hHApN6eTSp/view?usp=sharing) | [Depth Estimation Pre-trained Models](https://drive.google.com/file/d/1ptCS9YM5rdA1nXu3bTtdmLEqHa_87TTX/view?usp=sharing) |

* It also has the implementation our previous work: SuPer Framework. | [Website](https://sites.google.com/ucsd.edu/super-framework) | [Data](https://drive.google.com/file/d/1ZxWw2kNmgeMhBXAGyovL2icXHzn2OCVV/view?usp=sharing) |

## Setup the environmemt
### Option A: Using Docker
1. If you don't have CUDA support installed, install NVIDIA CUDA Driver and CUDA Toolkit by following [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#). We recommend using the [runfile](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-installation) installation method. The runfile we recommend is [version 12.2](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local).
2. Install [Docker](https://docs.docker.com/desktop/install/linux-install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) by following these two linked documents. If you have Docker already installed and have CUDA gpg key already set up, you can simply install NVIDIA Container Toolkit by running the following command:
```bash
sudo apt-get install -y nvidia-container-toolkit
```
3. Build docker image by running the following command:
```bash
docker build -f ./docker/super_docker.Dockerfile -t super ./
```
4. Activate the docker container by running: `docker run --runtime=nvidia --gpus all -v $(pwd):/workspace/ -v <your-path-to-data>:/data  -it super /bin/bash`
  - `-v $(pwd):/workspace/` mounts the Python-Super working repo under `/workspace` in the docker container.
  - `-v  <your-path-to-data>:/data` mounts the data storage directory under `/data`. You need to change `<your-path-to-data>` to the source path you want to mount.
5. Verify the container has CUDA support by running `nvidia-smi` in the docker container. You should see GPU information printed.
6. **Important note**. If your GPU doesn't support CUDA Toolkit 12.2, you need to do the following:
  - Go to [NVIDIA CUDA channel on DockerHub](https://hub.docker.com/r/nvidia/cuda/tags) and search for a tag that matches your GPU.
  - Change line 1 in [docker/super_docker.Dockerfile](./docker/super_docker.Dockerfile) to match the tag you found.
  - The docker base image tag's CUDA version needs to match that of your GPU. If not, you need to [uninstall CUDA driver and CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html?highlight=uninstall#uninstallation) then continue with step 1-5.


### Option B: Create the conda environmemt.
* Creat the conda environment with required packages using the following command:
```
conda env create -f resources/environment.yaml
```
Note: This command may take 7 hours due to package confliction.

## Run the code
### SuPer
```
python run_super.py \
--model_name super_run \
--data_dir [path to data] \
--tracking_gt_file rgb/left_pts.npy \
--load_depth \
--load_valid_mask \
--sf_point_plane --mesh_rot --mesh_arap
```
Notes: 
* ```--model_name```: The name of the folder that includes the tracking results.
* ```--tracking_gt_file``` is the file of tracking ground truth for evaluation, this file is usually saved under the same directory as the RGB images.
* ```--load_depth``` is called to load the precomputed depths.
* We also provide another deep learning depth estimation model [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo). To use it, replace ```--load_depth``` with ```--depth_model raft_stereo --pretrained_depth_checkpoint_dir ./depth/raft_core/weights/raft-pretrained.pth --dilate_invalid_kernel 50```.
* ```--sf_point_plane```: Point-to-plane ICP loss, ```--mesh_rot```: Rot loss, ```--mesh_arap```: As-rigid-as-possible loss.
* If ```--use_derived_gradient``` is set, the quaternions & translations will be optimized with Levenberg-Marquardt Algorithm and derived gradient, while if ```--use_derived_gradient``` is not used, the quaternions & translations will be optimized with Adam optimizer and PyTorch autograd.
* Use tensorboard to see tracking records, results, and visualizations.

### Semantic-SuPer:
  - Run Semantic-SuPer:
    - Trial3(SuPerV2-T1)：
    ```
    python run_semantic_super.py \
    --model_name semantic_super_superv2_trial3 \
    --data_dir /path/to/trial_3 \
    --tracking_gt_file rgb/trial_3_l_pts.npy \
    --mesh_step_size 32 --edge_ids 5 10 11 13 14 17 20 23 24 25 26 27 28 29 30 31 32 \
    --num_layers 50 --pretrained_encoder_checkpoint_dir /path/to/depth_checkpoint \
    --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir /path/to/depth_checkpoint --post_process \
    --load_seg --seg_dir seg/DeepLabV3+ \
    --sf_soft_seg_point_plane --mesh_rot --mesh_face --sf_bn_morph --render_loss
    ```

    - Trial4(SuPerV2-T2)：
    ```
    python run_semantic_super.py \
    --model_name semantic_super_superv2_trial4 \
    --data_dir /path/to/trial_4 \
    --tracking_gt_file rgb/trial_4_l_pts.npy \
    --mesh_step_size 32 --edge_ids 1 3 5 6 12 13 14 17 20 22 25 \
    --num_layers 50 --pretrained_encoder_checkpoint_dir /path/to/depth_checkpoint \
    --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir /path/to/depth_checkpoint --post_process \
    --load_seg --seg_dir seg/DeepLabV3+ \
    --sf_soft_seg_point_plane --mesh_rot --mesh_face --sf_bn_morph --render_loss
    ```

    - Trial8(SuPerV2-T3)：
    ```
    python run_semantic_super.py \
    --model_name semantic_super_superv2_trial8 \
    --data_dir /path/to/trial_8 \
    --tracking_gt_file rgb/trial_8_l_pts.npy \
    --mesh_step_size 25 --edge_ids 3 6 7 9 11 12 13 16 19 20 21 22 23 24 25 26 27 30 31 32 34 35 36 \
    --num_layers 50 --pretrained_encoder_checkpoint_dir /path/to/depth_checkpoint \
    --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir /path/to/depth_checkpoint --post_process \
    --load_seg --seg_dir seg/DeepLabV3+ \
    --sf_soft_seg_point_plane --mesh_rot --mesh_face --sf_bn_morph --render_loss
    ```

    - Trial9(SuPerV2-T4)：
    ```
    python run_semantic_super.py \
    --model_name semantic_super_superv2_trial9 \
    --data_dir /path/to/trial_9 \
    --tracking_gt_file rgb/trial_9_l_pts.npy \
    --mesh_step_size 18 --edge_ids 3 4 5 7 8 9 11 13 17 24 25 29 31 36 37 46 50 51 \
    --num_layers 50 --pretrained_encoder_checkpoint_dir /path/to/depth_checkpoint \
    --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir /path/to/depth_checkpoint --post_process \
    --load_seg --seg_dir seg/DeepLabV3+ \
    --sf_soft_seg_point_plane --mesh_rot --mesh_face --sf_bn_morph --render_loss
    ```

- Run SuPer(our baseline):
  - Trial3(SuPerV2-T1)：
  ```
  python run_semantic_super.py \
  --model_name super_face_superv2_trial3 \
  --method super \
  --data_dir /path/to/trial_3 \
  --tracking_gt_file rgb/trial_3_l_pts.npy \
  --mesh_step_size 32 --edge_ids 5 10 11 13 14 17 20 23 24 25 26 27 28 29 30 31 32 \
  --num_layers 50 --pretrained_encoder_checkpoint_dir /path/to/depth_checkpoint \
  --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir /path/to/depth_checkpoint --post_process \
  --sf_point_plane --mesh_rot --mesh_face
  ```

  - Trial4(SuPerV2-T2)：
  ```
  python run_semantic_super.py \
  --model_name super_face_superv2_trial4 \
  --method super \
  --data_dir /path/to/trial_4 \
  --tracking_gt_file rgb/trial_4_l_pts.npy \
  --mesh_step_size 32 --edge_ids 1 3 5 6 12 13 14 17 20 22 25 \
  --num_layers 50 --pretrained_encoder_checkpoint_dir /path/to/depth_checkpoint \
  --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir /path/to/depth_checkpoint --post_process \
  --sf_point_plane --mesh_rot --mesh_face
  ```
 
  - Trial8(SuPerV2-T3)：
  ```
  python run_semantic_super.py \
  --model_name super_face_superv2_trial8 \
  --method super \
  --data_dir /path/to/trial_8 \
  --tracking_gt_file rgb/trial_8_l_pts.npy \
  --mesh_step_size 25 --edge_ids 3 6 7 9 11 12 13 16 19 20 21 22 23 24 25 26 27 30 31 32 34 35 36 \
  --num_layers 50 --pretrained_encoder_checkpoint_dir /path/to/depth_checkpoint \
  --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir /path/to/depth_checkpoint --post_process \
  --sf_point_plane --mesh_rot --mesh_face
  ```
  
  - Trial9(SuPerV2-T4)：
  ```
  python run_semantic_super.py \
  --model_name super_face_superv2_trial9 \
  --method super \
  --data_dir /path/to/trial_9 \
  --tracking_gt_file rgb/trial_9_l_pts.npy \
  --mesh_step_size 18 --edge_ids 3 4 5 7 8 9 11 13 17 24 25 29 31 36 37 46 50 51 \
  --num_layers 50 --pretrained_encoder_checkpoint_dir /path/to/depth_checkpoint \
  --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir /path/to/depth_checkpoint --post_process \
  --sf_point_plane --mesh_rot --mesh_face
  ```

* Semantic-SuPer shares many parameters with SuPer, here we only introduce those that are different from SuPer.
* ```--mesh_step_size``` controls the step size used to initialize mesh grid. "To initialize the ED graph, since the depths vary a lot between trials, instead of using a fixed step size to generate the mesh, we choose the step size for each trial by ensuring the average edge length of the graph is around 5mm." 
* ```--edge_ids```: The ID of labeled points that are considered as edge points. This is used to report the reprojection errors for the edge points.
* We finetune [Monodepth2](https://github.com/nianticlabs/monodepth2) on Semantic-SuPer for depth estimation. Both the encoder and decoder code are from their official implementation. Our Monodepth2 checkpoints can be downloaded from [here](https://drive.google.com/file/d/1_8-TifbIlEegxKCjZpIfMn57sGCt7xHc/view?usp=share_link). Involved parameters: ```--num_layers```, ```--pretrained_encoder_checkpoint_dir```, ```--depth_model```, ```--pretrained_depth_checkpoint_dir```, ```--post_process```.
* We use the [Segmentation Models Pytorch (SMP) package](https://github.com/qubvel/segmentation_models.pytorch) to get semantic segmentation masks. ```--load_seg``` is called to load the precomputed semantic segmentation maps.
* ```--num_classes``` is the number of semantic classes, the Semantic-SuPer data has three classes: chicken, beef, and surgical tool.
* ```--sf_soft_seg_point_plane```: Semantic-aware point-to-plane ICP loss, ```--mesh_face```: Face loss, ```--sf_bn_morph```: Semantic-aware morphing loss, ```--render_loss``` (not in-use for now): Rendering loss.
* Ensure that you have the desired segmentation masks ready. You may produce them by modifying and running seg/inference.sh, using the pretrained checkpoints [here](https://drive.google.com/drive/folders/1qzv0KKo_t0VfVQkeNvbeB--4klCDXKiU?usp=sharing). Alternatively, you may train new checkpoints using the ground truths in the folders above and the seg/train.sh script.
* Use tensorboard to see tracking records, results, and visualizations.

## Tune the model.
The tracking performance can be influenced by:
1. Quality of input data (e.g., the depth map).
2. ```--mesh_step_size```: Grid step size to initialize the ED graph. ```--num_neighbors```, ```--num_ED_neighbors```: Number of neighboring surfels and ED nodes.
3. ```--th_dist```, ```--th_cosine_ang```: Thresholds to decide if two surfels could be merged.
4. ```--?_weight```: Weights for the losses.

## Results
* Reprojection errors:
  - SuPer V1:

  |model|commit id|[depth model] <br> Fine-tuned Mono2|[depth model] <br> pre-trained <br> RAFT-S|reproj. error <br> mean(std)|
  |---  |---  |---  |---  |---  |
  |super|8a5091c|:heavy_check_mark:|-|9.2(13.2)|
  |super|8a5091c|-|:heavy_check_mark:|11.5(16.3)|

  - Semantic SuPer Data:

  |model|commit id|[depth model] <br> Fine-tuned Mono2|[depth model] <br> pre-trained <br> RAFT-S|[data] <br> Trial1|[data] <br> Trial2|[data] <br> Trial3|[data] <br> Trial4|reproj. error <br> mean(std) <br> all pts, edge pts |
  |---  |---  |---  |---  |---  |---  |---  |---  |---  |
  |super|8a5091c|:heavy_check_mark:|-|:heavy_check_mark:|-|-|-|8.9, 11.1|
  |semantic-super|8a5091c|:heavy_check_mark:|-|:heavy_check_mark:|-|-|-|6.2, 6.5|
  |super|8a5091c|:heavy_check_mark:|-|-|:heavy_check_mark:|-|-|9.1, 10.6|
  |semantic-super|8a5091c|:heavy_check_mark:|-|-|:heavy_check_mark:|-|-|7.5, 8.6|
  |super|8a5091c|:heavy_check_mark:|-|-|-|:heavy_check_mark:|-|6.7, 7.3|
  |semantic-super|8a5091c|:heavy_check_mark:|-|-|-|:heavy_check_mark:|-|6.1, 6.6|
  |super|8a5091c|:heavy_check_mark:|-|-|-|-|:heavy_check_mark:|4.4, 5.2|
  |semantic-super|8a5091c|:heavy_check_mark:|-|-|-|-|:heavy_check_mark:|4.3, 5.0|

## Documentation
The documentation of the code is available [here](https://docs.google.com/document/d/1goWAr8oYvFmYdtoZNrDrbGh4bU2chzOkBxqeONUnIgc/edit?usp=sharing).

## Contact
shl102@ucsd.edu, k5wang@ucsd.edu
