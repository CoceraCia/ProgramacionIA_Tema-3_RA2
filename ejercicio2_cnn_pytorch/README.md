# Exercise 2 - CNN with PyTorch

Architecture: `Conv2d(1,16,3) -> ReLU -> MaxPool -> Conv2d(16,32,3) -> ReLU -> MaxPool -> Linear(800,128) -> ReLU -> Linear(128,10)`

- Optimizer: Adam (lr 0.001) · Loss: CrossEntropyLoss · 3 epochs, batch 64
- Evaluation on the 10,000 test images: accuracy + confusion matrix

Plots in `resultados/`:
- `filtros_conv1.png`: the 16 filters learned by the first conv layer
- `mapas_caracteristicas.png`: what conv1 and conv2 "see" for a test digit
- `matriz_confusion.png`

```bash
.venv/bin/python ejercicio2_cnn_pytorch/main.py
```
