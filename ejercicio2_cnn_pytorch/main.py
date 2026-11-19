# Base de datos: Dataset MNIST de dígitos escritos a mano
# Objetivo: Clasificar imágenes de dígitos del 0 al 9.
# Ejercicio 2: PyTorch
# Tarea: Construye una red neuronal convolucional (CNN) usando PyTorch para clasificar las imágenes del dataset MNIST.
# Requisitos:
#   Define la arquitectura de la CNN con capas convolucionales, capas de pooling y capas totalmente conectadas.
#   Usa un optimizador (por ejemplo, Adam) y una función de pérdida (por ejemplo, Cross-Entropy) para entrenar el modelo.
#   Visualiza las características aprendidas por las capas convolucionales utilizando una herramienta como matplotlib.
#   Evalúa el rendimiento del modelo usando métricas como la precisión (accuracy) y la matriz de confusión.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

HERE = Path(__file__).parent
RESULTADOS = HERE / "resultados"
EPOCAS = 3
torch.manual_seed(42)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 28x28 -> conv 3x3 -> 26x26 -> pool -> 13x13
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        # 13x13 -> conv 3x3 -> 11x11 -> pool -> 5x5
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 digitos
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # sin softmax, ya lo hace CrossEntropyLoss


def cargar_datos():
    # ToTensor pasa los pixeles de 0-255 a 0-1
    tf = transforms.ToTensor()
    train = datasets.MNIST(HERE / "datos", train=True, download=True, transform=tf)
    test = datasets.MNIST(HERE / "datos", train=False, download=True, transform=tf)
    print(f"Train: {len(train)} images | Test: {len(test)} images")
    return DataLoader(train, batch_size=64, shuffle=True), DataLoader(test, batch_size=1000), test


def entrenar(modelo, train_loader):
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        perdida_total = 0
        for imagenes, etiquetas in train_loader:
            optimizador.zero_grad()
            perdida = criterio(modelo(imagenes), etiquetas)
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item()
        print(f"Epoch {epoca}/{EPOCAS} - loss: {perdida_total / len(train_loader):.4f}")


def evaluar(modelo, test_loader):
    modelo.eval()
    reales, predichas = [], []
    with torch.no_grad():
        for imagenes, etiquetas in test_loader:
            predichas += modelo(imagenes).argmax(1).tolist()
            reales += etiquetas.tolist()
    acc = accuracy_score(reales, predichas)
    matriz = confusion_matrix(reales, predichas)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\n----- CONFUSION MATRIX -----")
    print(matriz)
    ConfusionMatrixDisplay(matriz, display_labels=range(10)).plot(cmap="Blues", colorbar=False)
    plt.title(f"CNN PyTorch - accuracy {acc:.2%}")
    plt.tight_layout()
    plt.savefig(RESULTADOS / "matriz_confusion.png", dpi=120)
    plt.close()


def visualizar(modelo, test):
    # 1) los 16 filtros 3x3 que ha aprendido la primera capa convolucional
    filtros = modelo.conv1.weight.detach()[:, 0]
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filtros[i], cmap="gray")
        ax.set_title(f"F{i + 1}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Conv1 filters (3x3)")
    plt.savefig(RESULTADOS / "filtros_conv1.png", dpi=120)
    plt.close()

    # 2) que "ve" la red: mapas de caracteristicas de un digito al pasar por conv1 y conv2
    imagen, etiqueta = test[0]
    with torch.no_grad():
        mapas1 = modelo.relu(modelo.conv1(imagen.unsqueeze(0)))[0]
        mapas2 = modelo.relu(modelo.conv2(modelo.pool(mapas1.unsqueeze(0))))[0]
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    axes[0, 0].imshow(imagen[0], cmap="gray")
    axes[0, 0].set_title(f"Input ({etiqueta})", fontsize=8)
    for i in range(1, 8):
        axes[0, i].imshow(mapas1[i - 1], cmap="viridis")
        axes[0, i].set_title(f"conv1 #{i}", fontsize=8)
    for i in range(16):
        ax = axes[1 + i // 8, i % 8]
        ax.imshow(mapas2[i], cmap="viridis")
        ax.set_title(f"conv2 #{i + 1}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Feature maps of one test digit")
    plt.savefig(RESULTADOS / "mapas_caracteristicas.png", dpi=120)
    plt.close()


def main():
    RESULTADOS.mkdir(exist_ok=True)
    train_loader, test_loader, test = cargar_datos()
    modelo = CNN()
    print(modelo)
    print("\n----- TRAINING -----")
    entrenar(modelo, train_loader)
    evaluar(modelo, test_loader)
    visualizar(modelo, test)
    print(f"\nSaved the plots in {RESULTADOS.name}/ ")


if __name__ == "__main__":
    main()
