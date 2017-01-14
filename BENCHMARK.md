# Experiments setup
+ CUDA 8.0
+ cuDNN 5.1
+ Titan X (Pascal)

# chainer

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

optimize kernels + no data grad
```
Average Forward:   17.4910337448  ms   
Average Backward:  30.8514846325  ms   
Average Total:     48.3425183773  ms   

```


### Googlenet
no tuning
```
Average Forward:   92.5838790894  ms  
Average Backward:  200.850806594  ms  
Average Total:     293.434685683  ms  

```

no data grad
```
Average Forward:   92.3806648254  ms     
Average Backward:  191.122313523  ms     
Average Total:     283.502978349  ms     

```


optimize kernels + no data grad
```
Average Forward:   90.8184089661  ms  
Average Backward:  181.053549671  ms  
Average Total:     271.871958637  ms  

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

