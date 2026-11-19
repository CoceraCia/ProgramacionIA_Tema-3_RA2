# Base de datos: Dataset MNIST de dígitos escritos a mano
# Objetivo: Clasificar imágenes de dígitos del 0 al 9.
# Ejercicio 1: Scikit-learn (Perceptrón Multicapa)
# Tarea: Construye un modelo de Perceptrón Multicapa (MLP) utilizando la clase MLPClassifier
# de Scikit-learn para clasificar las imágenes del dataset MNIST.
# Requisitos:
#   Preprocesa los datos: Normaliza los valores de los píxeles de las imágenes (dividiendo por 255).
#   Divide el dataset en conjuntos de entrenamiento y prueba (por ejemplo, 80% entrenamiento, 20% prueba).
#   Entrena el modelo con diferentes configuraciones de hiperparámetros (número de capas ocultas,
#   neuronas por capa, función de activación, optimizador, etc.)
#   Evalúa el rendimiento del modelo usando métricas como la precisión (accuracy) y la matriz de confusión.

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

HERE = Path(__file__).parent
RESULTADOS = HERE / "resultados"

# configuraciones que vamos a probar: (nombre, capas ocultas, activacion, optimizador)
CONFIGS = [
    ("A", (50,), "relu", "adam"),
    ("B", (128,), "relu", "adam"),
    ("C", (128, 64), "relu", "adam"),
    ("D", (128, 64), "tanh", "adam"),
    ("E", (128, 64), "relu", "sgd"),
]


def cargar_datos():
    print("Loading MNIST (the first time it downloads ~15 MB)...")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, data_home=HERE / "datos",
                        return_X_y=True, parser="liac-arff")  # el parser de pandas falla con pandas 3
    X = X / 255.0  # normalizamos los pixeles de 0-255 a 0-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {X_train.shape[0]} images | Test: {X_test.shape[0]} images | {X.shape[1]} pixels each")
    return X_train, X_test, y_train, y_test


def entrenar(config, X_train, X_test, y_train, y_test):
    nombre, capas, activacion, optimizador = config
    modelo = MLPClassifier(hidden_layer_sizes=capas, activation=activacion, solver=optimizador,
                           max_iter=20, random_state=42)
    inicio = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # con 20 epocas no "converge", ya lo sabemos
        modelo.fit(X_train, y_train)
    segundos = time.perf_counter() - inicio
    acc = accuracy_score(y_test, modelo.predict(X_test))
    print(f"{nombre}  layers={str(capas):<10} {activacion:<5} {optimizador:<5} accuracy={acc:.4f}  ({segundos:.0f} s)")
    return modelo, acc


def main():
    X_train, X_test, y_train, y_test = cargar_datos()
    RESULTADOS.mkdir(exist_ok=True)

    print("\n----- TRAINING 5 CONFIGURATIONS -----")
    resultados = [(c, *entrenar(c, X_train, X_test, y_train, y_test)) for c in CONFIGS]

    config, mejor, acc = max(resultados, key=lambda r: r[2])
    print(f"\nBest configuration: {config[0]} {config[1]} {config[2]} {config[3]} -> accuracy {acc:.4f}")

    y_pred = mejor.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred)
    print("\n----- CONFUSION MATRIX (best model) -----")
    print(matriz)

    disp = ConfusionMatrixDisplay(matriz, display_labels=mejor.classes_)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"MLP {config[1]} {config[2]} {config[3]} - accuracy {acc:.2%}")
    plt.tight_layout()
    plt.savefig(RESULTADOS / "matriz_confusion.png", dpi=120)
    print(f"\nSaved {RESULTADOS.name}/matriz_confusion.png")


if __name__ == "__main__":
    main()
