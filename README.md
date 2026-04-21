 "Self-Pruning Neural Network"

This repository contains the implementation of a Self-Pruning Neural Network for the Tredence Studio AI Engineering Internship Case Study.

The model is built using PyTorch and trained on the CIFAR-10 dataset. It utilizes a custom neural network layer (`PrunableLinear`) that dynamically learns to prune its own weakest connections during the training phase, adapting its architecture on the fly.

## Repository Structure

* `main.py`: The core executable script containing the `PrunableLinear` class, the `PrunableNet` model architecture, and the complete training/evaluation loop.
* `report.md`: A detailed report analyzing the L1 sparsity logic and comparing the trade-offs between accuracy and sparsity across different penalty (lambda) values.
* `requirements.txt`: The list of Python dependencies required to run the code.

## How to Run

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/vanshikaahluwalia86/Tredence-AI-Internship-Case-Study.git](https://github.com/vanshikaahluwalia86/Tredence-AI-Internship-Case-Study.git)
   cd Tredence-AI-Internship-Case-Study
