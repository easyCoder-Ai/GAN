# 📊 Evaluating GANs: Understanding GAN Evaluation with Fréchet Inception Distance (FID)

## 📜 **Project Overview**
Evaluating Generative Adversarial Networks (GANs) is a challenging task due to the absence of a straightforward performance metric. In this project, I explore the challenges of evaluating GANs and implement the Fréchet Inception Distance (FID)—a widely used metric to measure the quality of generated images.

This project was completed as part of a Coursera course on GANs, and I am publishing it here to showcase my implementation of GAN evaluation techniques.

## 🎯 **Learning Objectives**
By working through this project, I aimed to:
✅ Understand the challenges associated with evaluating GANs.

✅ Implement the Fréchet Inception Distance (FID) to measure GAN performance.

✅ Analyze how FID helps alleviate some common issues in GAN evaluation.

## 💡 **Why is GAN Evaluation Difficult?**
1️⃣ Loss is Uninformative: Unlike classifiers, GAN loss does not always indicate how realistic the generated images are. A low loss might mean the training has collapsed rather than improved.

2️⃣ No Clear Non-Human Metric: Since the goal of a GAN is to generate images that "look real," the best evaluation would involve human judgment. However, this is impractical, so we need alternative metrics like FID.

3️⃣ Measuring Realism and Diversity: A good GAN should generate diverse, high-quality images. Metrics like FID help quantify how similar the generated distribution is to the real one.

## 🔬 **What is Fréchet Inception Distance (FID)?**
FID compares the statistics (mean and covariance) of real and generated images in the feature space of a pre-trained model (typically Inception v3). The formula is:

$d(X, Y) = \Vert\mu_X-\mu_Y\Vert^2 + \mathrm{Tr}\left(\Sigma_X+\Sigma_Y - 2 \sqrt{\Sigma_X \Sigma_Y}\right)$

