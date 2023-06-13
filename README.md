# Neural Network Training Optimization using Cuckoo Search

This project is a comparison of traditional backpropagation method and Cuckoo Search algorithm for training a neural network. It aims to find a better and efficient way of training a neural network by optimizing the weights and biases.

## Overview

The problem of training a neural network is essentially to find the optimal set of weights and biases that minimize a loss function. In this project, we implemented a neural network from scratch and trained it using two methods:

- Traditional backpropagation
- Cuckoo Search optimization algorithm

We then compared the performance of the two methods.

## Dataset

The project uses the Iris dataset. The Iris dataset is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems". It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species.

## Libraries Used

- Numpy
- Scikit-learn
- Matplotlib

## Results

The project demonstrates that the Cuckoo Search algorithm can be used to effectively train a neural network. It is compared with the traditional backpropagation method and the performance of the models are evaluated based on metrics like accuracy, precision, recall, and F1 score. The confusion matrix is also used to evaluate the models.

## Usage

```python
python main.py
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- Inspired by the research paper "Cuckoo Search via Lévy flights" by Xin-She Yang and Suash Deb.

---

Please modify this README to fit the exact details and requirements of your project.
