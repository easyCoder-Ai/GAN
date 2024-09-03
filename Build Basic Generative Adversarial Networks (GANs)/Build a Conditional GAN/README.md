# Conditional GAN for Handwritten Digit Generation
This repository contains a project where I implemented a Conditional Generative Adversarial Network (cGAN) to generate hand-written images of digits from the MNIST dataset. Unlike traditional GANs, the cGAN allows us to condition the generation process on specific digit classes, enabling the generation of images of specific digits by inputting their corresponding class vectors.

## 📜 **Project Overview**
The goal of this project is to build and train a Conditional GAN (cGAN) to generate images of hand-written digits (0-9) conditioned on a specified class vector. This allows the generator to produce an image corresponding to a particular digit based on the input class.

In this project, we explore the use of noise vectors and class vectors in controlling the output of the GAN and visualize the impact these inputs have on the generated images.

## 📚 **Learning Objectives**
- Understand Conditional vs. Unconditional GANs: Learn how a conditional GAN differs from an unconditional GAN and the advantages of conditioning the generation process.
- Explore Class and Noise Vectors: Understand the roles of class vectors and noise vectors in the generation process and how they influence the output.
- Hands-on Implementation: Build the generator and discriminator networks with support for conditional inputs using the MNIST dataset.
Visualize Generation: Analyze and visualize how the cGAN generates specific digits and how the noise and class vectors contribute to the diversity of outputs.

## 🛠️ **Technologies Used**
- PyTorch: Primary framework used for building and training the Conditional GAN.
- Python: The programming language used throughout the project.
- Torchvision: Utilized for easy access to the MNIST dataset.
- Matplotlib: Used to visualize the generated images and training progress.

## 🚀 **How to Run the Code**
1.  Clone the Repository:
    - You can receive codes with:
        ```bash
        git clone https://github.com/easyCoder-Ai/GAN.git
        cd Build Basic Generative Adversarial Networks (GANs)/Build a Conditional GAN

2. Install Dependencies:
    - Make sure you have PyTorch and other necessary libraries installed:
    
            pip install -r requirements.txt -e <your current environment>

3.  Run 

    - You can run the code with:

        ```bash
        python3 main.py

## 📊 **Results**
As the training progresses, the Conditional GAN will generate images of digits conditioned on the input class. The images are visualized at different stages to observe how the quality and accuracy of the generated digits improve over time. The generator becomes better at producing digits that match the specified class while maintaining variety due to the noise vector.

## 💡 **Key Takeaways**
- Conditioning Enhances Control: By conditioning the GAN on a specific class, you gain control over the output, allowing you to generate specific types of images on demand.
- Vector Interplay: The class vector determines the digit to be generated, while the noise vector adds variability, resulting in diverse but class-specific images.
- GAN Flexibility: The principles learned here can be applied to generate other types of conditioned images, such as animals or faces, by adjusting the input data.