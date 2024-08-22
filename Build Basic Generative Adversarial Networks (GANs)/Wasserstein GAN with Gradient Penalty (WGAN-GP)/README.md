# Wasserstein GAN with Gradient Penalty (WGAN-GP) for Improved Stability
This repository contains my implementation of a Wasserstein GAN with Gradient Penalty (WGAN-GP). The WGAN-GP addresses some of the stability issues encountered with traditional GANs, particularly in preventing mode collapse and improving training dynamics. The project leverages PyTorch to build and train the model.

📜 Project Overview
The objective of this project is to implement a more stable version of a Generative Adversarial Network (GAN), known as the Wasserstein GAN with Gradient Penalty (WGAN-GP). Unlike standard GANs, which use a discriminator to classify images as real or fake, WGAN-GP uses a critic that scores images with real numbers. The project includes building the generator and critic networks, implementing the W-loss function, and applying gradient penalties to ensure the model's stability during training.

# Specifically, the project focuses on:
- Constructing the Generator and Critic networks.
- Implementing the Wasserstein loss (W-loss) function.
- Incorporating gradient penalties to prevent mode collapse.
- Training the WGAN-GP model on a dataset and generating realistic synthetic images.
- Visualizing the generated images throughout the training process to monitor progress.

📚 Learning Objectives
- Building a Stable GAN: Gain hands-on experience in constructing and training a Wasserstein GAN with Gradient Penalty (WGAN-GP) to overcome the limitations of traditional GANs.
- Training with W-loss: Understand and apply the Wasserstein loss function for training the GAN.
- Gradient Penalty Implementation: Learn to incorporate gradient penalties to stabilize the training process and prevent mode collapse.
- Model Training: Train the WGAN-GP model and observe its effectiveness in generating realistic images.
- Visualization: Track the progress of the GAN by visualizing the synthetic images generated during the training process.

🛠️ Technologies Used
- PyTorch: The primary deep learning framework used for building and training the WGAN-GP.
- Python: The programming language used throughout the project.
- Torchvision: For dataset management and preprocessing.
- Matplotlib: To visualize training progress and the generated images.

🚀 How to Run the Code

1. # Clone the Repository:

   git clone https://github.com/easyCoder-Ai/GAN.git
   cd Wasserstein GAN with Gradient Penalty (WGAN-GP)

2. # Install Dependencies:
    - Make sure you have PyTorch and other necessary libraries installed:
    
            pip install -r requirements.txt -e <your current environment>

3. # Run 

    python3 main.py


📊 Results
During training, the WGAN-GP model generates increasingly realistic images as the epochs progress. The inclusion of the gradient penalty significantly improves training stability, helping to prevent mode collapse and leading to more consistent quality in the generated images.

💡 Key Takeaways
- Improved Stability: The WGAN-GP effectively addresses some of the common issues with traditional GANs, particularly in terms of training stability and preventing mode collapse.
- Enhanced Learning: The use of the W-loss function and gradient penalties enables the model to learn more effectively, producing higher quality images.
- Hands-on Experience: Implementing and training the WGAN-GP model provides a deeper understanding of advanced GAN architectures and their benefits.

