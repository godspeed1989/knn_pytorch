# Pytorch KNN in CUDA
We calculate `distance matrix` and `topk indices` in Python.
The CUDA code just gathers the nearest neighbor points with `topk indices`.
## Install
```shell
cd knn_pytorch
make
python knn_pytorch.py
```
## Notes
- This repository works with pytorch 1.0
- Works with batched tensors
