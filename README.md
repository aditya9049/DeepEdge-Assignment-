# DeepEdge-Assignment-
# Pixel Coordinate Prediction using Deep Learning

## Problem Statement

Given a 50×50 grayscale image where:
- Exactly **one pixel** has value **255** (white)
- All other **2,499 pixels** have value **0** (black)

**Task**: Build a deep learning model to predict the (x, y) coordinates of that single bright pixel.

## Solution Approach


1. **Convolutional Neural Network (CNN)**
   - CNNs excel at spatial pattern recognition
   - Translation invariant - works regardless of pixel position
   - Efficient: fewer parameters than fully connected networks
   - Natural fit for image-based regression tasks

2. **Dataset Design**
   - **10,000 training samples**: Sufficient for CNN convergence
   - **Uniform random distribution**: Prevents positional bias
   - **Separate test set**: Ensures fair evaluation

3. **Smart Preprocessing**
   - Image normalization [0, 255] → [0, 1]
   - Coordinate normalization [0, 49] → [0, 1] for stable training
   - Improves gradient flow and convergence speed

4. **Architecture Choices**
   - 3 Conv blocks with increasing filters (16→32→64)
   - Batch normalization for training stability
   - Dropout for regularization
   - Global average pooling reduces parameters
   - Sigmoid activation for [0, 1] output range

