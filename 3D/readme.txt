export nnUNet_raw_data_base="/Hmsunet/DATASET/nnUNet_raw" 
export nnUNet_preprocessed="/Hmsunet/DATASET/nnUNet_preprocessed" 
export RESULTS_FOLDER="/Hmsunet/DATASET/nnUNet_trained_models"
export CUDA_VISIBLE_DEVICES=0


#Then first check the data format for errors and preprocess,
cd /Hmsunet/hmsunet/experiment_planning
python nnUNet_plan_and_preprocess.py -t 021 --verify_dataset_integrity

#train
cd /Hmsunet/hmsunet/run/
python run_training.py 3d_fullres nnUNetTrainerV2 Task021_Synapse 0 --npz

#predict
cd /Hmsunet/hmsunet/inference
python predict.py -i /Hmsunet/DATASET/nnUNet_raw/nnUNet_raw_data/Task021_Synapse/imagesTs/ -o /Hmsunet/DATASET/pos_models_out/outSynapse/ -f 0 -m /Hmsunet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task021_Synapse/nnUNetTrainerV2__nnUNetPlansv2.1/

# Refer to the nnUNet and nnFormer codes if you run into problems.