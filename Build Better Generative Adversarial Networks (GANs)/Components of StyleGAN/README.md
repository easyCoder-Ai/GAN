# 🧬 Implementing StyleGAN Components: Truncation Trick, Mapping Layers, AdaIN & More

## 📜 Project Overview

This notebook walks through the implementation of key components that make StyleGAN a groundbreaking architecture in the field of generative models. StyleGAN introduced several innovations over traditional GANs, enabling more control over the image generation process and significantly improving output quality.

You'll gain hands-on experience with truncation trick, mapping networks, noise injection, adaptive instance normalization (AdaIN), and progressive growing, which are foundational to StyleGAN’s architecture.

This project is part of the Coursera Build Better Generative Adversarial Networks (GANs) specialization. The goal is to deepen understanding by rebuilding the internal mechanisms of StyleGAN from scratch..

---

## 🎯 Learning Objectives

✅ Understand the differences between StyleGAN and traditional GANs.

✅ Implement and experiment with:

- Truncation Trick

- apping Networks

- Noise Injection

- Adaptive Instance Normalization (AdaIN)

- Progressive Growing of GANs

✅ Visualize intermediate results to see how each component contributes to the overall generation pipeline.

---

## 🧱 Components of StyleGAN

- Truncation Trick
Controls the diversity vs. fidelity trade-off in the latent space.

- Mapping Layer
Separates the latent space (z) from the intermediate latent space (w) to improve disentanglement.

- Noise Injection
Adds stochastic variation at different levels of the generator to improve realism.

- AdaIN (Adaptive Instance Normalization)
Modulates style at different layers of the generator using features from the mapping network.

- Progressive Growing
Trains the generator and discriminator from low to high resolution, stabilizing training and improving quality.


---



## 🛠️ Technologies Used

- 🔹 Python – Core scripting language
- 🔹 PyTorch – Used for implementing GANs and classifiers
- 🔹 Torchvision – For dataset handling and image transformations
- 🔹 Matplotlib – Visualization of generated images and classifier outputs

---

## 🧠 Insights and Observations

- Noise injection introduces small, high-frequency features that dramatically increase visual realism.

- Truncation trick helps balance diversity and quality, especially in faces and high-resolution samples.

- Progressive growing leads to faster convergence and improved stability.

---



## 🚀 **How to Run the Code**
1.  Clone the Repository:
    - You can receive codes with:
        ```bash
        git clone https://github.com/easyCoder-Ai/GAN.git
        cd Build Better Generative Adversarial Networks (GANs)/Components of StyleGAN

2. Install Dependencies:
    - Make sure you have PyTorch and other necessary libraries installed:
    
            pip install -r requirements.txt -e <your current environment>

3.  Run 

    - You can run the code with:

        ```bash
        python3 main.py


--- 

## 📊 Results

- ✅ Implemented each StyleGAN component modularly and tested its effect on generated outputs.
- ✅ Visualized the role of AdaIN and noise injection in modifying image attributes.
- ✅ Gained intuition for how architectural choices affect generation control and image fidelity.

---

## 💡 Key Takeaways

- StyleGAN’s design encourages style-based control, allowing for manipulation of specific visual attributes.

- Each architectural tweak—while subtle in isolation—adds up to massive gains in quality and usability.

- Understanding these inner components equips you to build custom GAN variants with more interpretability and control.
