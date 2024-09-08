# Controllable GAN Generation Using Classifier Gradients
This repository contains a project focused on implementing a controllable generation technique using Generative Adversarial Networks (GANs). The method allows for the manipulation of generated images by adjusting the generator's input vectors (z-vectors) based on gradients from a pre-trained classifier.

## 📜 **Project Overview**
The objective of this project is to explore controllable image generation by leveraging a pre-trained GAN and classifier. By utilizing gradients from the classifier, the generator's input can be adjusted to emphasize or de-emphasize specific features in the generated images. This project primarily focuses on manipulating images from the CelebA dataset, a large-scale dataset of celebrity images with various annotations.

## Specifically, the project focuses on:
- Controllability Method Implementation: Using classifier gradients to modify the generator’s input vectors to control the features in the generated images.
- Pre-trained Model Utilization: Working with a pre-trained generator and classifier to emphasize the controllability aspects without the need to train these models from scratch.
- Feature Manipulation: Observing the effects of feature manipulation on generated images and addressing challenges posed by entangled features.
- Dataset: Utilizing the CelebA dataset for generating and manipulating high-quality, colored images.

## 📚 **Learning Objectives**
- Understanding Controllability in GANs: Learn how to manipulate GAN outputs by adjusting input vectors using gradients from a classifier.
- Feature Manipulation Techniques: Develop skills to enhance or suppress specific features in generated images by leveraging a pre-trained classifier.
- Overcoming Challenges: Address and resolve issues related to entangled features, which can complicate the controllability process.
- Hands-on Practice: Gain practical experience in using GANs for controllable image generation, with a focus on real-world datasets.

## 🛠️ **Technologies Used**
- PyTorch: The deep learning framework used for implementing and training the GAN and classifier.
- Python: The primary programming language used in the project.
- CelebA Dataset: A dataset of celebrity images used for training and testing the controllability method.
- Matplotlib: For visualizing the changes in generated images during the controllability process.


## 🚀 **How to Run the Code**
1.  Clone the Repository:
    - You can receive codes with:
        ```bash
        git clone https://github.com/easyCoder-Ai/GAN.git
        cd Build Basic Generative Adversarial Networks (GANs)/Controllable Generation

2. Install Dependencies:
    - Make sure you have PyTorch and other necessary libraries installed:
    
            pip install -r requirements.txt -e <your current environment>

3.  Run 

    - You can run the code with:

        ```bash
        python3 main.py



## 📊 **Results**
The project demonstrates the effectiveness of controllable generation using GANs and classifier gradients. By manipulating the input vectors based on classifier feedback, the GAN is able to generate images with enhanced or suppressed features as desired. The CelebA dataset provides a rich variety of features for experimentation, and the project showcases how these features can be manipulated in a controllable manner.

## 💡 **Key Takeaways**
- Controllable Generation: Successfully using classifier gradients to control specific features in GAN-generated images.
- Real-world Applications: This technique has potential applications in various domains, including content creation, data augmentation, and more.
- Challenges in Controllability: The project highlights the challenges of feature entanglement and provides insights into how they can be managed.
