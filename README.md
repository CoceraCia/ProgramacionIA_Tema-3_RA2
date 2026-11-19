# Programación IA - Tema 3 (RA2): MNIST

Handwritten digit classification (0-9) with the MNIST dataset using three different libraries.

| Folder | Exercise |
|---|---|
| `ejercicio1_mlp_sklearn` | Multilayer Perceptron with scikit-learn (`MLPClassifier`), 5 hyperparameter configurations |
| `ejercicio2_cnn_pytorch` | CNN with PyTorch + learned filters and feature maps |
| `ejercicio3_cnn_keras` | CNN with the Keras Sequential API + TensorBoard |

MNIST is downloaded automatically the first time (into each `datos/` folder, not uploaded). Plots are saved in each `resultados/` folder.

## Setup

The `.venv` is already created with everything installed. To create it again (Python 3.12):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` installs the CPU version of PyTorch (lighter). Everything runs fine on CPU.

## Run

```bash
.venv/bin/python ejercicio1_mlp_sklearn/main.py
.venv/bin/python ejercicio2_cnn_pytorch/main.py
.venv/bin/python ejercicio3_cnn_keras/main.py
```
