import argparse
import time

import numpy as np
import cupy

from chainer import cuda
from chainer import optimizers
import chainer
import chainer.functions as F


workspace_size = 1 * 2 ** 30  # 8GB
chainer.cuda.set_max_workspace_size(workspace_size)

niter = 2


configs = [
    #{'ni': 3, 'no': 96, 'k': 11, 'iw': 128, 'ih': 128, 'bs': 128, 'd': 1},
    {'ni': 64, 'no': 128, 'k': 9, 'iw': 64, 'ih': 64, 'bs': 8, 'd': 1},
    #{'ni': 128, 'no': 128, 'k': 9, 'iw': 32, 'ih': 32, 'bs': 128, 'd': 1},
]

for config in configs:
    print config
    conv = F.Convolution2D(
        config['ni'], config['no'], config['k'], config['d']).to_gpu()
    input_ = chainer.Variable(cupy.random.uniform(
        size=(config['bs'], config['ni'], config['ih'], config['iw']),
        dtype=np.float32))

    
    time.sleep(2.)
    cupy.cuda.Stream(null=True).synchronize()

    for i in range(niter):
        time.sleep(1.)
        start = cupy.cuda.Event()
        end = cupy.cuda.Event()
        start.record()
        out = conv(input_)
        end.record()
        end.synchronize()
        forward_time = cupy.cuda.get_elapsed_time(start, end)
        

        time.sleep(0.5)
        out.grad = cupy.random.uniform(size=out.shape, dtype=np.float32)
        cupy.cuda.Stream(null=True).synchronize()
        time.sleep(0.5)
        start = cupy.cuda.Event()
        end = cupy.cuda.Event()
        start.record()
        out.backward()
        end.record()
        end.synchronize()
        backward_time = cupy.cuda.get_elapsed_time(start, end)
        if i > 0:
            print 'forward time {}   backward time {}'.format(
                forward_time, backward_time)
