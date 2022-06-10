## Dataset - Perk

### DP

n_gpu: 4  
batch_size: 64  
eval_step: 20  
time: 2:05:22  
acc: 91.7

### DDP

n_gpu: 4  
batch_size: 64  
eval_step: 20  
time: 0:33:28  
acc: 91.1


n_gpu: 4  
batch_size: 128  
eval_step: 10  
time: 0:17:29  
acc: 91.1


### DDP mixed precision

n_gpu: 2  
batch_size: 64  
eval_step: 20  
time: 0:17:45  
acc: 92.4

n_gpu: 4  
batch_size: 64  
eval_step: 20  
time: 0:18:46(Somehow slower)  
acc: 92.7

n_gpu: 4  
batch_size: 128  
eval_step: 10  
time: 0:14:50  
acc: 0.918  
GPU Memory: 21901 MB (vs 23957 MB w/o mixed precision)

### Quantization

model size: 1.17 GB (vs 2.2 GB)

batch_size: 50  
acc: 92.2 (vs 92.2)  
runtime_per_batch: 2.84s/it (vs 3.15it/s)


batch_size: 1  
acc: 93.3  (vs 92.2)  
runtime_per_batch: 9.21it/s (vs 6.12it/s)

## Dataset Noun

### DP

n_gpu: 4  
batch_size: 300  
epoch: 1  
eval_step: 6  
time: 0:27:45  
mcc: 94.7

### DDP

n_gpu: 4  
batch_size: 600  
epoch: 2  
eval_step: 4  
time: 0:20:05  
mcc: 96.2

## MT5

2.2 GB

Quantized: 