# sim
A small image model implementation

## How to run
```
wget https://systemds.apache.org/assets/datasets/mnist/train-images-idx3-ubyte.gz
wget https://systemds.apache.org/assets/datasets/mnist/train-labels-idx1-ubyte.gz
gzip -d train-*.gz
make run -C gpu -j 6
```