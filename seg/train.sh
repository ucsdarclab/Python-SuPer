#!/bin/bash

model="smp_unet18"

epochs=50
workers=0

train_batch_size=16 
val_batch_size=16 
full_res_validation="True"

lr=1e-3
optimizer="Adam"
wd=0.00001
lr_steps=4
step_gamma=0.1
dice_loss_factor=0.0
focal_loss_factor=0.0

resized_height=480 # for random crop
resized_width=640 # for random crop
RandomCropSize=480640
cropSize=-1

dataset="super"
data_dir="src/data/datasets/pig9"
json_path="src/data/classes/pigClasses.json"
#checkpoint_path="/home/arcseg/jonathan/ARCSeg/results/resnet18_unet/cholec/bs_32_lr1e-3_e100/resnet18_unet_cholec_bs32lr0.001e100_checkpoint"

display_samples="False"
save_samples="True"

	save_dir="results/${model}/${dataset}_super/${RandomCropSize}/Full_Data/bs_train${train_batch_size}_val${val_batch_size}/imsize_${resized_height}x${resized_width}_wd_${wd}_optim_${optimizer}_lr${lr}_steps_${lr_steps}_gamma_${step_gamma}/e${epochs}_seed6210"
	seg_save_dir="${save_dir}/seg_results"

	mkdir -p $save_dir

	python src/trainSuper.py \
	    --model $model \
	    --workers $workers \
	    --trainBatchSize $train_batch_size \
	    --valBatchSize $val_batch_size \
	    --full_res_validation $full_res_validation \
	    --resizedHeight $resized_height \
	    --resizedWidth $resized_width \
	    --cropSize $cropSize \
	    --lr $lr \
	    --dice_loss_factor $dice_loss_factor \
	    --focal_loss_factor $focal_loss_factor \
	    --epochs $epochs \
	    --lr_steps $lr_steps \
	    --step_gamma $step_gamma \
	    --optimizer $optimizer \
	    --wd $wd \
	    --dataset $dataset \
	    --display_samples $display_samples \
	    --save_samples $save_samples \
	    --data_dir $data_dir \
	    --json_path $json_path \
	    --save_dir $save_dir \
	    --seg_save_dir $seg_save_dir \
	    |& tee -a "${save_dir}/debug.log"
