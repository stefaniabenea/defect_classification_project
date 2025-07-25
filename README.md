# NEU Surface Defect Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) model to classify surface defects in industrial images. The dataset used is the NEU Surface Defect Database, which contains images of six defect types.

The pipeline includes:
- Data loading and augmentation (using Albumentations)
- CNN model training with PyTorch
- Evaluation metrics and visualization (loss, accuracy, confusion matrix)
- Visualization of learned filters and activations
- Inference script for new images or folders

## Setup

### Requirements
- Python 3.8+
- PyTorch
- torchvision
- albumentations
- matplotlib
- pandas
- scikit-learn
- pillow
- numpy

You can install dependencies with:

```bash
pip install -r requirements.txt

## Dataset Structure
The dataset folder should be organized as follows:

data/
├── train/
│   └── images/
│       ├── class_1/
│       ├── class_2/
│       ├── ...
├── validation/
│   └── images/
│       ├── class_1/
│       ├── class_2/
│       ├── ...

## Usage

### Training
Run the training script to train the CNN model with data augmentation, batch normalization, dropout, and learning rate scheduling.

### Visualization
- **Filters**: Visualize convolutional filters of specified layers.
- **Activations**: Visualize activations of specific layers for a given input image.
- **Augmentations**: View sample augmented images.

Example command line usage for visualization:

```bash
python visualize.py --mode filters --layer conv1
python visualize.py --mode augmentation --img_path path/to/image.jpg
python visualize.py --mode activations --layer conv2 --img_path path/to/image.jpg

You can specify multiple modes at once:
python visualize.py --mode filters activations --layer conv1 --img_path path/to/image.jpg

## Inference
Use the inference script to predict labels for a single image or all images in a folder. Results are saved to a CSV file.

```bash
python inference.py --img_path path/to/image.jpg
python inference.py --folder_path path/to/images/

## Project Structure
.
├── data/                  # Dataset folders
├── models/                # Saved models
├── plots/                 # Training plots and confusion matrix
├── utils.py               # Utility functions for transforms, plotting, visualization
├── model.py               # CNN model definition
├── train.py               # Training script
├── visualize.py           # Visualization script with argparse
├── inference.py           # Inference script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

