# CIFAR-10 Image Classification with VTL

A CNN example for CIFAR-10 image classification using the V Tensor Library (VTL).

## CI note

Default GitHub Actions **does not compile** this example (OOM on runners). PR CI uses
`nn_cifar10_tiny_synth` instead. For full-tree validation, add the PR label **`full-ml`**
or run locally: `VJOBS=1 ~/.vmodules/vtl/bin/test --full`.

## Overview

This example demonstrates:
- Building a CNN with VTL layers (Conv2D, MaxPool, Flatten, Linear)
- Training with mini-batches using cross-entropy loss
- Using the Adam optimizer
- Evaluating accuracy on train and validation sets

## Architecture

The CNN follows this architecture:

```
Input: [batch, 3, 32, 32]
├── Conv2D (3→64, 3×3) + ReLU + MaxPool (2×2)
├── Conv2D (64→128, 3×3) + ReLU + MaxPool (2×2)
├── Conv2D (128→256, 3×3) + ReLU + MaxPool (2×2)
├── Flatten
├── Linear (4096→256) + ReLU
├── Linear (256→10) + Softmax
```

After 3 MaxPool layers with stride 2: 32 → 16 → 8 → 4
So the final feature map is 256 channels × 4 × 4 = 4096 features.

## CIFAR-10 Dataset

The CIFAR-10 dataset consists of:
- 50,000 training images (5 batches of 10,000)
- 10,000 test images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Image size: 32×32×3 RGB

The dataset is automatically downloaded from the University of Toronto repository on first run.

## Running the Example

```bash
cd examples/nn_cifar10
v run .
```

Or from the vtl root:

```bash
v run examples/nn_cifar10
```

## Configuration

You can modify the training parameters in `main.v`:

```v
const batch_size = 64 // Mini-batch size
const epochs = 10 // Number of training epochs
const learning_rate = 0.001 // Adam learning rate
```

## Expected Output

The training progress shows:
- Per-batch loss and accuracy during training
- Epoch-level train loss and accuracy
- Validation loss and accuracy after each epoch

Example output:
```
Loading CIFAR-10 dataset...
Dataset loaded successfully!
Training samples: 50000
Test samples: 10000
Image shape: [3, 32, 32]

Building CNN model...
Model built successfully!
Training for 10 epochs with batch size 64

========================================
Epoch 1/10
========================================
  Batch 0/781: Loss=2.3026, Acc=9.38%
  Batch 100/781: Loss=2.0456, Acc=32.81%
  ...
  Epoch 1 Summary:
    Train Loss: 1.8234
    Train Accuracy: 45.23%

  Running validation...
    Val Loss: 1.6523
    Val Accuracy: 52.15%
```

## Files

- `main.v` - Main training script with data loading, training loop, and evaluation
- `model.v` - CNN architecture definition
- `data.v` - CIFAR-10 dataset loading utilities
- `README.md` - This file

## Notes

- Training takes significant time on CPU. For GPU acceleration (CUDA/Vulkan),
  ensure VTL is compiled with GPU support.
- The first run will download ~170MB of CIFAR-10 data.
- Memory usage is approximately 2-4GB depending on batch size.