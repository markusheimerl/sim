# sim
A small image model implementation

This project implements an autoregressive image generation model using a transformer architecture. The model learns to generate images pixel by pixel, treating image generation as a sequence prediction problem similar to language modeling.

The architecture operates on 28×28 grayscale images from the MNIST dataset, processing each image as a sequence of 784 pixel values. To enable class-conditional generation, the model embeds class information directly into the first pixel of each image during training. This allows the model to learn which digit class it should generate.

The model begins with a token embedding layer that maps each pixel value (0-255) to a continuous vector representation. These embeddings are then augmented with learned 2D positional encodings that provide spatial information about each pixel's row and column position in the image. Unlike standard 1D positional encodings used in text models, the 2D encoding splits the embedding dimension in half, dedicating one half to encoding row position and the other half to encoding column position using sinusoidal functions.

The core of the architecture is a multi-layer transformer that processes the embedded pixel sequences. Each transformer layer consists of causal self-attention and feed-forward networks with residual connections. The causal attention mechanism ensures that when predicting each pixel, the model can only attend to previously generated pixels, maintaining the autoregressive property necessary for sequential generation. This is crucial for image generation, as it forces the model to generate pixels in raster-scan order (left-to-right, top-to-bottom).

After processing through all transformer layers, an output projection layer maps the final hidden states to logits over all 256 possible pixel values. During training, the model learns to predict each pixel given all previous pixels using cross-entropy loss. During generation, the model samples pixels one at a time from the predicted probability distribution, building up the complete image sequentially.

The training process uses the AdamW optimizer, which combines adaptive learning rates with weight decay regularization. The optimizer maintains exponential moving averages of both gradients and squared gradients, using these to adapt the learning rate for each parameter individually while applying L2 regularization to encourage smaller weights and better generalization.

For generation, the model starts with only the class label embedded in the first pixel, then autoregressively samples each subsequent pixel from the learned probability distribution. Temperature scaling controls the randomness of the sampling process, with lower temperatures producing more deterministic outputs and higher temperatures increasing diversity.

The implementation uses BLAS (Basic Linear Algebra Subprograms) for efficient matrix operations, allowing the model to train effectively on modern hardware.

## How to run
```
wget https://systemds.apache.org/assets/datasets/mnist/train-images-idx3-ubyte.gz
wget https://systemds.apache.org/assets/datasets/mnist/train-labels-idx1-ubyte.gz
gzip -d train-*.gz
make run -j 6
```