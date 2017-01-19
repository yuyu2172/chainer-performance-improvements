# Experiments setup
+ CUDA 8.0
+ cuDNN 5.1
+ Titan X (Pascal)

# chainer

### alexnet

workspace size (8MB)
```
alexnet
Average Forward:   22.2445697784  ms
Average Backward:  40.9176445007  ms
Average Total:     63.1622142792  ms
```

Results below all set workspace size as 1GB.

no tuning
```
alexnet                                 
Average Forward:   18.23636606  ms      
Average Backward:  41.731580019  ms     
Average Total:     59.967946079  ms     
```

no data grad, no softmax, no sgd update
```
alexnet
Average Forward:   18.2486175537  ms
Average Backward:  34.547221756  ms
Average Total:     52.7958393097  ms
```

no data grad, no softmax, no sgd update, remove unnecessary `x._grad+=gx`
```
alexnet
Average Forward:   18.1039072037  ms
Average Backward:  32.1488834381  ms
Average Total:     50.2527906418  ms
```


no data grad, no softmax, no sgd update, remove unnecessary `x._grad+=gx`, optimize conv kernels

```
Average Forward:   17.5481090546  ms
Average Backward:  28.2826019287  ms
Average Total:     45.8307109833  ms
```


# Torch

### alexnet
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

### googlenet
```
ModelType: GoogleNet    Kernels: cudnn  Input shape: 128x3x224x224     
cudnn                                   :updateOutput():      75.69    
cudnn                                :updateGradInput():      98.16    
cudnn                              :accGradParameters():      70.89    
cudnn                                          :Forward:      75.69    
cudnn                                         :Backward:     169.05    
cudnn                                            :TOTAL:     244.74    
```

