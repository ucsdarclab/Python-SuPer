# Python-SuPer

## Create the conda environmemt.
* Creat the conda environment with required packages using the following command:
```
conda env create -f resources/env.yml
```
## Prepare the data.
* Put RGB and depth images into your data directory.
* Put *eva_id.npy* and *labelPts.npy*, which can be found in *./labels*, into your data directory. *labelPts.npy* includes the ground truth positions of the 20 tracked points in the SuPer dataset.
* Update the datase directory (variable name: *data_dir*) in *super.sh*.
## Run the code
* Run the code using the following command:
```
./super.sh
```
* A .npy file that consists of tracking results will be saved in the same folder when main.py ends. Run evaluate.py to read current and previous tracking results, and plot the tracking performance:
```
python evaluate.py
```
