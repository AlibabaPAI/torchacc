# Quick Start

We use ResNet-50 as an example to demonstrate how to accelerate the training with `TorchAcc`.

## Torch Native Task

Below is the code for an ResNet-50 in Torch:

```python
import time
import torch
import torchvision

batch_size = 64
log_steps = 20
inputs = torch.randn(6400, 3, 224, 224)
labels = torch.randint(0, 100, (6400,))
dataset = torch.utils.data.TensorDataset(inputs, labels)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = torchvision.models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(4):
    model.train()
    start_time = time.time()
    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % log_steps == 0:
            iteration_time = time.time() - start_time
            throughputs = batch_size * log_steps / iteration_time
            print(f'Epoch: {epoch}, Step: {i}, Loss: {loss:.4f}, Throughputs: {throughputs:.4f} samples/s')
            start_time = time.time()
```

The results on an `80G A100` are as follows:

```shell
$ python resnet_native.py

Epoch: 0, Step: 20, Loss: 5.0722, Throughputs: 146.9960 samples/s
Epoch: 0, Step: 40, Loss: 5.6211, Throughputs: 933.5072 samples/s
Epoch: 0, Step: 60, Loss: 5.0938, Throughputs: 933.4178 samples/s
Epoch: 0, Step: 80, Loss: 4.8861, Throughputs: 931.6142 samples/s
Epoch: 0, Step: 100, Loss: 4.8116, Throughputs: 927.7186 samples/s
Epoch: 1, Step: 20, Loss: 4.6499, Throughputs: 777.6132 samples/s
Epoch: 1, Step: 40, Loss: 4.7558, Throughputs: 929.3011 samples/s
Epoch: 1, Step: 60, Loss: 4.6438, Throughputs: 923.1462 samples/s
Epoch: 1, Step: 80, Loss: 4.6413, Throughputs: 933.8570 samples/s
Epoch: 1, Step: 100, Loss: 4.7834, Throughputs: 934.5286 samples/s
```

## Single GPU Acceleration with `TorchAcc`

By modifying 3 lines of code, you can call TorchAcc's `accelerate` interface to accelerate model training:

```diff
  import time
  import torch
+ import torchacc
  import torchvision

  batch_size = 64
  log_steps = 20
  inputs = torch.randn(6400, 3, 224, 224)
  labels = torch.randint(0, 100, (6400,))
  dataset = torch.utils.data.TensorDataset(inputs, labels)
  train_loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=True, num_workers=4)

  model = torchvision.models.resnet50()
  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, 100)

- device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
- model = model.to(device)
+ model, train_loader = torchacc.accelerate(model, train_loader)
+ device = model.device
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

  for epoch in range(2):
      model.train()
      start_time = time.time()
      for i, (inputs, labels) in enumerate(train_loader, 1):
-         inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)

          loss.backward()
          optimizer.step()
          if i % log_steps == 0:
              iteration_time = time.time() - start_time
              throughputs = batch_size * log_steps / iteration_time
              print(f'Epoch: {epoch}, Step: {i}, Loss: {loss:.4f}, Throughputs: {throughputs:.4f} samples/s')
              start_time = time.time()
```

The main changes include:
* Remove the original `model to device` logic, and use `torchacc.accelerate` to wrap the model and dataloader;
* Remove `batch to device`, as the dataloader wrapper will automatically handle it.

The results with TorchAcc enabled are as follows:

```bash
$ python resnet_acc.py

Epoch: 0, Step: 20, Loss: 5.3560, Throughputs: 38.8063 samples/s
Epoch: 0, Step: 40, Loss: 4.8361, Throughputs: 1144.4567 samples/s
Epoch: 0, Step: 60, Loss: 4.8088, Throughputs: 1141.0203 samples/s
Epoch: 0, Step: 80, Loss: 4.6511, Throughputs: 1131.4401 samples/s
Epoch: 0, Step: 100, Loss: 4.6082, Throughputs: 1191.8075 samples/s
Epoch: 1, Step: 20, Loss: 4.6110, Throughputs: 722.4752 samples/s
Epoch: 1, Step: 40, Loss: 4.6275, Throughputs: 1150.5421 samples/s
Epoch: 1, Step: 60, Loss: 4.6875, Throughputs: 1163.3742 samples/s
Epoch: 1, Step: 80, Loss: 4.6159, Throughputs: 1176.7777 samples/s
Epoch: 1, Step: 100, Loss: 4.6067, Throughputs: 1184.0109 samples/s
```

It can be observed that the model undergoes compilation optimization at the start of training. After the compilation is completed, the average iteration throughput shows a 16% improvement compared to the native Torch (average throughput improvement after step 40).

## Multiple GPUs Acceleration with `TorchAcc`

### Data Parallel

No modifications needed, just replace the execution command with `torchrun`. `TorchAcc` will automatically detect the number of GPUs and default to data parallel training.

To better and more accurately view the output information, we can have only rank 0 print the output and multiply the throughputs by the number of GPUs to calculate the global throughputs. The code can be modified as follows:

```diff
  import time
  import torch
  import torchacc
  import torchvision

  batch_size = 64
  log_steps = 20
  inputs = torch.randn(6400, 3, 224, 224)
  labels = torch.randint(0, 100, (6400,))
  dataset = torch.utils.data.TensorDataset(inputs, labels)
  train_loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=True, num_workers=4)

  model = torchvision.models.resnet50()
  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, 100)

  model, train_loader = torchacc.accelerate(model, train_loader)
  device = model.device
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

  for epoch in range(2):
      model.train()
      start_time = time.time()
      for i, (inputs, labels) in enumerate(train_loader, 1):
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)

          loss.backward()
          optimizer.step()
-         if i % log_steps == 0:
+         if i % log_steps == 0 and torch.distributed.get_rank() == 0:
              iteration_time = time.time() - start_time
-             throughputs = batch_size * log_steps / iteration_time
+             throughputs = batch_size * log_steps / iteration_time * torch.distributed.get_world_size()
              print(f'Epoch: {epoch}, Step: {i}, Loss: {loss:.4f}, Throughputs: {throughputs:.4f} samples/s')
              start_time = time.time()
```

```bash
$ torchrun --nproc_per_node=4 resnet_acc.py

Epoch: 0, Step: 20, Loss: 4.8700, Throughputs: 311.2281 samples/s
Epoch: 0, Step: 40, Loss: 4.7842, Throughputs: 4359.9072 samples/s
Epoch: 0, Step: 60, Loss: 4.6687, Throughputs: 4370.7357 samples/s
Epoch: 0, Step: 80, Loss: 4.6690, Throughputs: 4385.6592 samples/s
Epoch: 0, Step: 100, Loss: 4.7295, Throughputs: 4540.9381 samples/s
Epoch: 1, Step: 20, Loss: 4.6978, Throughputs: 2852.0285 samples/s
Epoch: 1, Step: 40, Loss: 4.6760, Throughputs: 4378.8057 samples/s
Epoch: 1, Step: 60, Loss: 4.6696, Throughputs: 3888.6148 samples/s
Epoch: 1, Step: 80, Loss: 4.6757, Throughputs: 4347.2658 samples/s
Epoch: 1, Step: 100, Loss: 4.6497, Throughputs: 4421.5528 samples/s
```

### FSDP (Fully Sharded Data Paarlell)

You only need to configure the TorchAcc `Config` and pass it to the `torchacc.accelerate` function to easily achieve FSDP training.

```diff
  import time
  import torch
  import torchacc
  import torchvision

  batch_size = 64
  log_steps = 20
  inputs = torch.randn(6400, 3, 224, 224)
  labels = torch.randint(0, 100, (6400,))
  dataset = torch.utils.data.TensorDataset(inputs, labels)
  train_loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=True, num_workers=4)

  model = torchvision.models.resnet50()
  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, 100)

+ config = torchacc.Config()
+ config.dist.fsdp.size = 4
+ config.dist.fsdp.wrap_layer_cls = {"Bottleneck"}
+ model, train_loader = torchacc.accelerate(model, train_loader, config)
  device = model.device

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

  for epoch in range(2):
      model.train()
      start_time = time.time()
      for i, (inputs, labels) in enumerate(train_loader, 1):
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          if i % log_steps == 0:
              iteration_time = time.time() - start_time
              throughputs = batch_size * log_steps / iteration_time
              print(f'Epoch: {epoch}, Step: {i}, Loss: {loss:.4f}, Throughputs: {throughputs:.4f} samples/s')
              start_time = time.time()

```

The shell command for running FSDP tasks is the same as data parallelism:

```bash
$ torchrun --nproc_per_node=4 resnet_acc.py
```