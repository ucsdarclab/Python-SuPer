# Python-SuPer

## Create the conda environmemt.
* Creat the conda environment with required packages using the following command:
```
conda env create -f resources/env.yml
```
## Prepare the data.
* Put RGB and depth images into your data directory.
* Put eva_id.npy and labelPts.npy, which can be found in labels, into your data directory. eva_id.npy includes the ground truth, SURF and C++ SuPer tracking results.
* Update the datase directory in main.sh.
## Run the code
* Run the code using the following command:
```
./main.sh
```
* A .npy file that consists of tracking results will be saved in the same folder when main.py ends. Run evaluate.py to read current and previous tracking results, and plot the tracking performance:
```
python evaluate.py
```
## Document:
* inputStream.py
  - Read data
  - Preprocess depth map
* nodes.py: Class handles surfels and ED nodes.
* LM.py: Implementation of Levenberg–Marquardt algorithm in Pytorch.
* evaluate.py: Plot tracking results of the 20 labeled points in the SuPer dataset.
* pt_matching.py: Feature matching
* utils.py
* render.py: Visualization
* config.py: Configuration parameters
  
## TODOs:
### Speed up the code:
* Try better feature matching methods. Current OpenCV-based corr loss actually degrade the tracking performance.
* Try available Pytorch KNN
  - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/knn.html
* Faster solver for system of linear equations.

### Next version of SuPer:
* Spatial transformer.
* Replace SIFT feature matching with advanced matching methods:
  - https://github.com/PruneTruong/DenseMatching
  - https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark
* Include evaluation code.
* Improve visualization: 1) Visualize surfels; 2) heatmap-style (better words?) display.
