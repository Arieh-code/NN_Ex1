import numpy as np
import seaborn as sns
from numpy import random
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS
from mlxtend.classifier import Adaline
from mlxtend.plotting import plot_decision_regions


def partC(X, y, classifier, condition):
    print("Condition " + condition + " || Data = 1,000 || Learning rate: 1/10 || n = 100 || Hidden layers = [8, 2]")
    print("Score of correct prediction: ", classifier.score(X, y) * 100, "%")

    for layer in range(2, classifier.n_layers_):
        layer_i = forward(classifier, X, layer)
        if layer == classifier.n_layers_ - 1:
            geometric_diagram(classifier, X, layer - 1, layer_i, True)
        else:
            geometric_diagram(classifier, X, layer - 1, layer_i)
    # https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
    # used this site to understand decision region
    # Plotting decision regions
    plot_decision_regions(X, y.astype(int), clf=classifier, legend=2)

    # Adding axes annotations
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('MLP on condition ' + condition)
    plt.show()

    # confusion_matrix
    cm = confusion_matrix(classifier.predict(X), y)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("Confusion matrix - Data-set " + condition)
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()


def partD(X, y, classifier, condition):
    print("Condition " + condition + " || Data = 1,000 || Learning rate: 1/10 || n = 100 || Hidden layers = [8, 2]")

    last_hidden_layer = []
    for layer in range(1, classifier.n_layers_):
        layer_i = forward(classifier, X, layer)
        if layer == classifier.n_layers_ - 1:
            last_hidden_layer = layer_i
        geometric_diagram(classifier, X, layer - 1, layer_i)

    X_Adaline = np.array([last_hidden_layer[0], last_hidden_layer[1]]).T

    y[y < 0] = 0
    classifier_adaline = Adaline(epochs=2, eta=0.01, random_seed=0).fit(X_Adaline, y.astype(int))
    predict_adaline = classifier_adaline.predict(X_Adaline)

    print("Score of correct prediction: ", classifier_adaline.score(X_Adaline, y) * 100, "%")

    # Geometric diagram of the combination between MLP and Adaline Algorithms
    plt.scatter(x=X[predict_adaline == 0, 1], y=X[predict_adaline == 0, 0],
                alpha=0.9, c='red',
                marker='s', label=-1.0)

    plt.scatter(x=X[predict_adaline == 1, 1], y=X[predict_adaline == 1, 0],
                alpha=0.9, c='blue',
                marker='x', label=1.0)

    plt.title("Part D: MLP && Adaline Algorithms (Condition " + condition + ")")
    plt.legend(loc='upper left')
    plt.show()

    # confusion_matrix
    cm = confusion_matrix(predict_adaline, y)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("Confusion matrix - Data-set " + condition)
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()


def Build_Data(d_size, condition, n):
    data = np.empty((d_size, 2), dtype=object)
    random.seed(10)
    if condition == "A":
        for i in range(d_size):
            data[i, 0] = (random.randint(-10000, 10000) / n)
            data[i, 1] = (random.randint(-10000, 10000) / n)
    else:
        for i in range(d_size):
            data[i, 0] = (random.randint(-4000, 4000) / 1000)
            data[i, 1] = (random.randint(-4000, 4000) / 1000)

    train = np.zeros(d_size)

    if condition == "A":
        for i in range(d_size):
            if data[i][1] <= 1:
                train[i] = -1
            else:
                train[i] = 1

    if condition == "B":
        for i in range(d_size):
            if 4 <= (data[i][1] ** 2 + data[i][0] ** 2) <= 9:
                train[i] = 1
            else:
                train[i] = -1

    X = data.astype(np.float64)  # test
    y = train.astype(np.float64)  # train

    return X, y


def geometric_diagram(classifier, X, index_layer, layer, output=False):
    """
    layer[1] - input
    layer[2: n - 1] - hidden layers
    layer[n] - output
    :param classifier:
    :param X:
    :param index_layer:
    :param layer:
    :param output:
    :return:
    """
    index_n = 0
    for neuron in layer:
        plt.scatter(x=X[neuron == -1, 1], y=X[neuron == -1, 0],
                    alpha=0.9, c='red',
                    marker='s', label=-1.0)

        plt.scatter(x=X[neuron == 1, 1], y=X[neuron == 1, 0],
                    alpha=0.9, c='blue',
                    marker='x', label=1.0)

        plt.legend(loc='upper left')
        plt.title("Layer: " + str(index_layer) + " Neuron: " + str(index_n))
        plt.show()
        index_n += 1

    if output:
        # presentation of output layers
        output_layer = forward(classifier, X)
        print("output layer")
        plt.scatter(x=X[output_layer == -1, 1], y=X[output_layer == -1, 0],
                    alpha=0.9, c='red',
                    marker='s', label=-1.0)

        plt.scatter(x=X[output_layer == 1, 1], y=X[output_layer == 1, 0],
                    alpha=0.9, c='blue',
                    marker='x', label=1.0)

        plt.legend(loc='upper left')
        plt.title("Layer: " + str(index_layer + 1) + " Neuron: " + str(0))
        plt.show()


def forward(classifier, X, layers=None):
    if layers is None or layers == 0:
        layers = classifier.n_layers_

    # Initialize first layer
    activation = X

    # Forward propagate
    hidden_activation = ACTIVATIONS[classifier.activation]
    for i in range(layers - 1):
        weight_i, bias_i = classifier.coefs_[i], classifier.intercepts_[i]
        activation = (activation @ weight_i) + bias_i
        if i != layers - 2:
            hidden_activation(activation)
    if activation.shape[1] > 1:
        ans = []
        for j in range(activation.shape[1]):
            ans.append(classifier._label_binarizer.inverse_transform(activation[:, j]))
        return ans
    hidden_activation(activation)
    return classifier._label_binarizer.inverse_transform(activation)


if __name__ == '__main__':
    d_size = 1000
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    # we used the site above to understand better the MLP classifier.S

    # Part C && D (condition A)
    X_a, y_a = Build_Data(d_size, "A", 100)

    MLP_Data_A = MLPClassifier(activation='logistic', learning_rate_init=0.1,
                               hidden_layer_sizes=(8, 2), random_state=1, max_iter=240).fit(X_a, y_a)
    partC(X_a, y_a, MLP_Data_A, "A")
    partD(X_a, y_a, MLP_Data_A, "A")

    # # Part C && D (condition B)

    X, y = Build_Data(d_size, "B", 100)

    MLP_Data_B = MLPClassifier(activation='logistic', learning_rate_init=0.1,
                               hidden_layer_sizes=(8, 2), random_state=1, max_iter=240).fit(X, y)
    partC(X, y, MLP_Data_B, "B")
    partD(X, y, MLP_Data_B, "B")
