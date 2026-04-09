Axon
Axon is a custom deep learning engine built directly on top of PyTorch tensors.

Rather than using high-level abstractions, this library implements the core logic of modern architectures from the ground up to ensure total transparency and control over the underlying mathematics.

🚀 Key Features
Custom Convolutional Layers: Implemented conv2D and conv2d_general with custom kernel shapes and manual spatial unfolding.

Attention Mechanisms: Full implementation of Multi-Headed Attention (MHA) and specialized attention blocks for Transformer-style architectures.

Normalization: Custom Batch Normalization (1D & 2D) and Layer Normalization.

Modular MLP: Flexible Multi-Layer Perceptron builder with automated parameter tracking.

🧠 Philosophy
This library is a "statue of mastery." It lacks the traditional nn.Module weight saving and quantization because it is designed to show the raw logic of how data moves through a network. While these features can be handled manually, the primary goal is deep conceptual understanding.

I will continue to update this engine as I explore and master new concepts in Deep Learning.
