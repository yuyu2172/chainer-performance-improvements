# Chainer Performance
This repository contains improvements I made on performance of Chainer. I observed improvements based on benchmark scripts at https://github.com/soumith/convnet-benchmarks.


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
4. fix code to stop calling `x._grad += gx` when `x._grad` is zero matrix. 


# Benchmarks
You can check a benchmark results at `Benchmarks.md`.


Final results on Alex net
### Chainer
```
Average Forward:   17.5475772858  ms  
Average Backward:  26.7493019104  ms  
Average Total:     44.2968791962  ms  
```

### Torch
```
Running on device: TITAN X (Pascal)
ModelType: AlexNet      Kernels: cudnn  Input shape: 128x3x224x224
cudnn                                   :updateOutput():      17.44
cudnn                                :updateGradInput():      14.29
cudnn                              :accGradParameters():      14.62
cudnn                                          :Forward:      17.44
cudnn                                         :Backward:      28.91
cudnn                                            :TOTAL:      46.35

```
