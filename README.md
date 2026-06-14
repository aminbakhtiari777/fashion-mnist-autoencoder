# Fashion MNIST Autoencoder

## Project Overview

This project implements a simple Autoencoder using the Fashion MNIST dataset.

The goal is to learn how neural networks can compress image data into a smaller latent representation and then reconstruct the original image.

## Dataset

Fashion MNIST Dataset

The dataset contains grayscale clothing images with size 28x28 pixels.

- Training images: 60,000
- Testing images: 10,000
- Image size: 28x28
- Flattened input size: 784

## What is an Autoencoder?

An Autoencoder is a neural network that learns to reconstruct its input.

The model receives an image as input and tries to output the same image.

The network has two main parts:

- Encoder
- Decoder

## Encoder

The Encoder compresses the original image into a smaller representation.

Architecture:

```text
784 → 128 → 64 → 32
