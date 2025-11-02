# Alzheimer's Classification Task 8

This folder contains the code for Task 8: Classifying Alzheimers. 

## Files

### Dataset
Dataset.py contains code that handles data loading, preprocessing and splitting.

### Modules
Modules.py contains code that define the model in this case, ConvNeXt. This code is obtained from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L87.

### Train
Train.py contains code for the main training script. 

## How it Works

1. **Preparing the Dataset**
- Two folders (AD - Alzheimers and NC - Normal Control) are loaded in from Rangpur provided data. 
- Data is split, ensuring no overlap, into train, validation and test sets. 

2. **Model**
- ConvNeXt for 2-class classification. 

3. **Training**
- Cross Entropy loss 
- AdamW optimiser
- Plot across epochs to view accuracy and loss
- Optimal learning rate scheduler

## How to Run
### Local
python train.py

### UQ Rangpur
sbatch run_task8.slurm
