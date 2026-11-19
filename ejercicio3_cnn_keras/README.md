# Exercise 3 - CNN with TensorFlow/Keras

Sequential model: `Conv2D(32) -> MaxPooling2D -> Conv2D(64) -> MaxPooling2D -> Flatten -> Dense(128) -> Dense(10, softmax)`

- Optimizer: adam · Loss: sparse_categorical_crossentropy · Metric: accuracy
- 5 epochs, batch 64, 10% of train used for validation
- Test accuracy + confusion matrix

Plots in `resultados/`: `perdida_entrenamiento.png` (loss and accuracy per epoch) and `matriz_confusion.png`.

## TensorBoard

The training logs are saved in `logs/`. To see the loss curves:

```bash
.venv/bin/tensorboard --logdir ejercicio3_cnn_keras/logs
```

and open http://localhost:6006
