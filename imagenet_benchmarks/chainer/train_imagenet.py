#!/usr/bin/env python
import argparse
import time

import numpy as np

from chainer import cuda
from chainer import optimizers

parser = argparse.ArgumentParser(
    description=' convnet benchmarks on imagenet')
parser.add_argument('--arch', '-a', default='alexnet',
                    help='Convnet architecture \
                    (alex, googlenet, vgga, overfeat)')
parser.add_argument('--batchsize', '-B', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

# Prepare model
print(args.arch)
if args.arch == 'alexnet':
    import alex
    model = alex.Alex()
elif args.arch == 'googlenet':
    import googlenet
    model = googlenet.GoogLeNet()
elif args.arch == 'vgga':
    import vgga
    model = vgga.vgga()
elif args.arch == 'overfeat':
    import overfeat
    model = overfeat.overfeat()
else:
    raise ValueError('Invalid architecture name')

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)

workspace_size = int(1 * 2**30)
import chainer

chainer.cuda.set_max_workspace_size(workspace_size)


def train_loop():
    # Trainer
    data = np.ndarray((args.batchsize, 3, model.insize, model.insize), dtype=np.float32)
    data.fill(33333)
    total_forward = 0
    total_backward = 0
    niter = 13
    n_dry = 3

    label = np.ndarray((args.batchsize), dtype=np.int32)
    label.fill(1)
    count = 0
    for i in range(niter):
        # print "Iteration", i

        x = xp.asarray(data)
        y = xp.asarray(label)
        
        #time.sleep(0.5)
        optimizer.zero_grads()
        start = xp.cuda.Event()
        end = xp.cuda.Event()
        start.record()
        loss, accuracy = model.forward(x, y)
        end.record()
        end.synchronize()
        time_ = xp.cuda.get_elapsed_time(start, end)
        if i > n_dry- 1:
            count += 1
            total_forward += time_
        # print "Forward step time elapsed:", time_, " ms"

        #time.sleep(0.5)
        start = xp.cuda.Event()
        end = xp.cuda.Event()
        start.record()
        loss.backward()
        end.record()
        end.synchronize()
        time_ = xp.cuda.get_elapsed_time(start, end)
        if i > n_dry - 1:
            total_backward += time_
        # print "Backward step time elapsed:", time_, " ms"

        ##time.sleep(0.5)
        #start = xp.cuda.Event()
        #end = xp.cuda.Event()
        #start.record()
        #optimizer.update()
        #end.record()
        #end.synchronize()
        #time_ = xp.cuda.get_elapsed_time(start, end)
        #if i > n_dry - 1:
        #    total_backward += time_
        ## print "Optimizer update time elapsed:", time_, " ms"

        del loss, accuracy
    print "Average Forward:  ", total_forward  / count, " ms"
    print "Average Backward: ", total_backward / count, " ms"
    print "Average Total:    ", (total_forward + total_backward) / count, " ms"
    print ""

train_loop()
