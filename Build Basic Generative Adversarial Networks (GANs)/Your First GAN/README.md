# Generative Adversarial Network (GAN) for Handwritten Digits

This repository contains my Coursera GAN specialization exam project, where I built and trained a Generative Adversarial Network (GAN) to generate hand-written images of digits (0-9) using PyTorch.

## 📜 **Project Overview**
The goal of this project is to create a GAN capable of generating realistic images of handwritten digits. The model is built from scratch using PyTorch, a popular deep learning framework.

Specifically, the project focuses on:
- Building both the Generator and Discriminator components of the GAN.
- Defining the loss functions required to train the GAN.
- Training the GAN and visualizing the generated images after each epoch.

## 📚 **Learning Objectives**
1. Building Components: Construct the Generator and Discriminator networks from scratch using PyTorch.
2. Loss Functions: Implement the appropriate loss functions to guide the GAN training process.
3. Training the GAN: Train the GAN using a dataset of handwritten digits (e.g., MNIST), and generate synthetic images.
4. Visualization: Visualize the images generated by the GAN to track progress during training.

## 🛠️ **Technologies Used**
- PyTorch: The deep learning framework used for building and training the GAN.
- Python: The core programming language for the project.


# 🚀 **How to Run the Code**
1.  Clone the Repository:
    - You can receive codes with:
        ```bash
        git clone https://github.com/easyCoder-Ai/GAN.git
        cd Build Basic Generative Adversarial Networks (GANs)/Your First GAN

2. Install Dependencies:
    - Make sure you have PyTorch and other necessary libraries installed:
    
            pip install -r requirements.txt -e <your current environment>

3.  Run 

    - You can run the code with:

        ```bash
        python3 main.py

## 🎯 **Results**
 
Throughout the training process, the generated images of digits are visualized to track the GAN's progress. After successful training, the GAN can generate synthetic handwritten digits that closely resemble real ones.