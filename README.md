# EP41 — Autoencoders & Variational Autoencoders (VAE)

> **Deep Learning Series · Episode 41 of 72 · Module 8 — Generative Deep Learning**

A plain autoencoder compresses and reconstructs, but its latent space is full of holes.  
A **VAE** fixes that with a probabilistic latent space + the reparameterization trick.  
We train one in pure PyTorch and generate brand-new MNIST digits.

## What you will learn

- Encoder → bottleneck → decoder + reconstruction loss
- Why a plain autoencoder can’t generate (the gaps problem)
- VAE’s probabilistic latent (μ, σ²)
- KL divergence — regularizing the latent space
- The reparameterization trick (`z = μ + σ ⊙ ε`)
- Full PyTorch implementation on MNIST
- Generating new digits + latent-space interpolation

## Key formulas
