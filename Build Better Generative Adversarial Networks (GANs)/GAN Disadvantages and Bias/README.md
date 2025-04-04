# 🔍 Detecting Bias in GANs: Analyzing Implicit Associations in Generated Data

## 📜 Project Overview

Generative Adversarial Networks (GANs) can unintentionally learn and amplify biases present in training data. In this project, I explore a technique for identifying potential biases in a GAN’s output by using a classifier to analyze correlations between generated features and protected attributes.

This notebook is part of the Coursera **Build Better Generative Adversarial Networks (GANs)** specialization. I’m publishing this to share my implementation and understanding of fairness and bias detection in generative models.

---

## 🎯 Learning Objectives

- ✅ Understand key fairness concepts: **Demographic Parity**, **Equality of Odds**, and **Equality of Opportunity**.
- ✅ Use a **classifier** to detect potential **biases** in GAN-generated data.
- ✅ Gain insight into the **challenges of maintaining fairness** in image generation while preserving diversity in protected class attributes.

---

## ⚖️ Why is Bias Detection Important in GANs?

- **GANs Learn from Biased Data**: If the training data contains social or demographic biases, the generator may replicate and even amplify those patterns.
- **Fairness is Complex**: Protected classes (e.g., gender, race) are social constructs, and defining fairness requires context and nuance.
- **Generators Should Still Represent Diversity**: Ideally, a GAN should produce images across all protected class categories without correlating unrelated features (like facial expressions or lighting) to those classes.

---

## 🧠 Approach to Bias Detection

- Use a **pre-trained classifier** to evaluate whether generated images unintentionally associate certain features with a protected class.
- Analyze the classifier’s outputs to assess if certain traits (e.g., smiling) are disproportionately present in one demographic group over another.
- This **post-hoc analysis** provides a proxy for detecting unfair associations without needing explicit conditioning on protected attributes.

---

## 🔬 Challenges

- ⚠️ **Protected Class Ambiguity**: The definition of what features belong to a class is not always clear-cut.
- ⚠️ **Classifier Limitations**: The classifiers used to detect bias may themselves be biased or inaccurate.
- ⚠️ **Balancing Diversity and Fairness**: We want generators to reflect real-world diversity without introducing stereotypical or skewed representations.

---

## 🛠️ Technologies Used

- 🔹 Python – Core scripting language
- 🔹 PyTorch – Used for implementing GANs and classifiers
- 🔹 Torchvision – For dataset handling and image transformations
- 🔹 Matplotlib – Visualization of generated images and classifier outputs

---

## 🚀 **How to Run the Code**
1.  Clone the Repository:
    - You can receive codes with:
        ```bash
        git clone https://github.com/easyCoder-Ai/GAN.git
        cd Build Better Generative Adversarial Networks (GANs)/GAN Disadvantages and Bias

2. Install Dependencies:
    - Make sure you have PyTorch and other necessary libraries installed:
    
            pip install -r requirements.txt -e <your current environment>

3.  Run 

    - You can run the code with:

        ```bash
        python3 main.py


--- 

## 📊 Results

- ✅ Used a classifier to evaluate fairness in GAN-generated samples.
- ✅ Identified possible correlations between protected attributes and visual features.
- ✅ Reflected on how GANs can inadvertently reinforce societal stereotypes.

---

## 💡 Key Takeaways

- GANs can encode and reflect social biases—even when not explicitly conditioned on protected attributes.
- Using classifiers to analyze generated data is a practical method for surface-level bias detection.
- Fairness in generative modeling is a nuanced challenge that requires careful evaluation and design.
