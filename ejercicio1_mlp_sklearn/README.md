# Exercise 1 - MLP with scikit-learn

- Loads MNIST (70,000 images of 28x28 = 784 pixels) with `fetch_openml`
- Normalizes the pixels (divided by 255)
- Splits 80% train / 20% test (stratified)
- Trains `MLPClassifier` with 5 configurations (hidden layers, neurons, activation, optimizer), 20 epochs each
- Shows the accuracy of each one and the confusion matrix of the best (`resultados/matriz_confusion.png`)

```bash
.venv/bin/python ejercicio1_mlp_sklearn/main.py
```

It takes a few minutes (it trains 5 models).
