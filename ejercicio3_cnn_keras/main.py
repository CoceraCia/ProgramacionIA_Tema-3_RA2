# Base de datos: Dataset MNIST de dígitos escritos a mano
# Objetivo: Clasificar imágenes de dígitos del 0 al 9.
# Ejercicio 3: TensorFlow/Keras
# Tarea: Construye una red neuronal convolucional (CNN) usando TensorFlow/Keras para clasificar las imágenes del dataset MNIST.
# Requisitos:
#   Utiliza la API secuencial de Keras para definir la arquitectura de la CNN (capas convolucionales,
#   capas de pooling y capas densas).
#   Define la función de pérdida, el optimizador y las métricas de evaluación (por ejemplo, accuracy y matriz de confusión).
#   Entrena el modelo y evalúa su rendimiento en el conjunto de prueba.
#   Puedes utilizar TensorBoard para visualizar la pérdida del modelo durante el entrenamiento.

import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # que TensorFlow no llene la consola de avisos

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

HERE = Path(__file__).parent
RESULTADOS = HERE / "resultados"
LOGS = HERE / "logs"
EPOCAS = 5
keras.utils.set_random_seed(42)


def cargar_datos():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # normalizamos a 0-1 y añadimos el canal (28, 28) -> (28, 28, 1)
    X_train = (X_train / 255.0)[..., np.newaxis]
    X_test = (X_test / 255.0)[..., np.newaxis]
    print(f"Train: {len(X_train)} images | Test: {len(X_test)} images")
    return X_train, y_train, X_test, y_test


def crear_modelo():
    modelo = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    modelo.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",  # etiquetas como numeros 0-9
                   metrics=["accuracy"])
    return modelo


def grafica_perdida(historial):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epocas = range(1, EPOCAS + 1)
    ax1.plot(epocas, historial.history["loss"], "o-", label="train")
    ax1.plot(epocas, historial.history["val_loss"], "o-", label="validation")
    ax1.set_title("Loss")
    ax2.plot(epocas, historial.history["accuracy"], "o-", label="train")
    ax2.plot(epocas, historial.history["val_accuracy"], "o-", label="validation")
    ax2.set_title("Accuracy")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.set_xticks(list(epocas))
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTADOS / "perdida_entrenamiento.png", dpi=120)
    plt.close()


def main():
    RESULTADOS.mkdir(exist_ok=True)
    X_train, y_train, X_test, y_test = cargar_datos()
    modelo = crear_modelo()
    modelo.summary()

    print("\n----- TRAINING -----")
    # TensorBoard guarda la perdida de cada epoca en logs/ -> tensorboard --logdir logs
    tensorboard = keras.callbacks.TensorBoard(log_dir=str(LOGS))
    historial = modelo.fit(X_train, y_train, epochs=EPOCAS, batch_size=64,
                           validation_split=0.1, callbacks=[tensorboard], verbose=2)
    grafica_perdida(historial)

    perdida, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {perdida:.4f} | Test accuracy: {acc:.4f}")

    y_pred = modelo.predict(X_test, verbose=0).argmax(axis=1)
    matriz = confusion_matrix(y_test, y_pred)
    print("\n----- CONFUSION MATRIX -----")
    print(matriz)
    ConfusionMatrixDisplay(matriz, display_labels=range(10)).plot(cmap="Blues", colorbar=False)
    plt.title(f"CNN Keras - accuracy {acc:.2%}")
    plt.tight_layout()
    plt.savefig(RESULTADOS / "matriz_confusion.png", dpi=120)
    plt.close()
    print(f"\nSaved the plots in {RESULTADOS.name}/ | TensorBoard: .venv/bin/tensorboard --logdir "
          f"{LOGS.relative_to(HERE.parent)}")
    print(f"TensorFlow {tf.__version__} ")


if __name__ == "__main__":
    main()
