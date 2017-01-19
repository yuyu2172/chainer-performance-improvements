# Chainer Performance
This repository contains improvements I made on performance of Chainer.


# How to get started

```
git submodule update --remote
cd chainer
pip install -e .

cd ../imagenet_benchmarks/chainer
python train_imagenet.py
```


# Things I have done
1. Run `cudnnFindConvolution*` to evaluate convolution algorithms at warmup, and use the fastest algorithm validated empirically.
2. Stop doing unnecessary gradient computation at bottom conv layer (a trick used by Torch benchmark script)
3. match computation burden to the Torch Benchmark (stop optimizer update, no softmax)


# TODOs

1. x._grad += gx

# Benchmarks
You can check a benchmark results at `BENCHMARK.md`.
