# Experiments setup
+ CUDA 8.0
+ cuDNN 5.1
+ Titan X (Pascal)

# chainer

# alexnet
no tuning
```
alexnet                                 
Average Forward:   18.23636606  ms      
Average Backward:  41.731580019  ms     
Average Total:     59.967946079  ms     
```

no data grad
```
alexnet                                
Average Forward:   18.1751487732  ms   
Average Backward:  34.9425219774  ms   
Average Total:     53.1176707506  ms   
```

no data grad, optimize conv kernels
```
Average Forward:   17.4910337448  ms   
Average Backward:  30.8514846325  ms   
Average Total:     48.3425183773  ms   

```

optimize conv kernels, no data grad, no softmax

```
Average Forward:   17.3968765259  ms 
Average Backward:  30.5509248734  ms 
Average Total:     47.9478013992  ms 
```

optimize conv kernels, no data grad, no softmax, no sgd update

```
Average Forward:   17.5450880051  ms    
Average Backward:  28.8155040741  ms    
Average Total:     46.3605920792  ms    
```

optimize conv kernels, no data grad, no softmax, no sgd update, remove unnecessary `x._grad+=gx`,

```
Average Forward:   17.5475772858  ms  
Average Backward:  26.7493019104  ms  
Average Total:     44.2968791962  ms  
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

