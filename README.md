# qiutong-xu
code for Pytorch Training

# Neurofibroma Detection and Classification (PyTorch)

This repository contains the PyTorch training code developed for the thesis project **“Mobile App for Detecting and Classifying Neurofibromas using Deep Learning”**, conducted at the University of Queensland (UQ).

The project aims to train an efficient convolutional neural network (CNN) model to detect and classify Neurofibromas (NF) from medical or dermatological images, and deploy the trained model on a mobile platform for on-device inference.

---

## Project Overview

- **Objective:**  
  Develop and train a lightweight deep learning model (EfficientNet-B0) capable of identifying *Neurofibroma (NF)*, *Non-NF*, and *Other* categories from image data.

- **Frameworks Used:**  
  - Python 3.10  
  - PyTorch 2.x  
  - Torchvision  
  - NumPy / Pandas / Matplotlib  
  - scikit-learn  
  - OpenCV  

- **Key Features:**
  - Dataset preprocessing with augmentation (rotation, flipping, contrast).
  - Fine-tuned **EfficientNet-B0** architecture for lightweight deployment.
  - **Focal Loss** for handling class imbalance.
  - Model export using **TorchScript Lite** for mobile integration.
  - Reproducible training pipeline with configurable hyperparameters.

---

## Folder Structure

Below is the recommended structure of the project repository:

nf_project/
├── data/ # dataset (not uploaded; use your own or ISIC/Radiopaedia)
├── models/ # saved model checkpoints (.pt or .pth)
├── src/ # main source code
│ ├── train_nf.py # main training script
│ ├── dataset.py # data loading and augmentation
│ ├── model.py # EfficientNet-B0 architecture
│ ├── loss.py # focal loss function
│ ├── utils.py # helper functions (metrics, plotting)
│ └── inference.py # inference/testing on new images
├── requirements.txt # Python dependencies
├── README.md
└── LICENSE


---

##  Installation

```bash
# clone the repo
git clone https://github.com/qiutong0908/qiutong-xu.git
cd qiutong-xu

# create virtual environment
python3 -m venv nf_env
source nf_env/bin/activate    # (on macOS/Linux)
# or: .\nf_env\Scripts\activate (on Windows)

# install dependencies
pip install -r requirements.txt



