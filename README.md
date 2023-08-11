# Project
### What is Machine Learning?
- Machine learning is a subset of AI where computers learn from data without being explicitly programmed. It involves training models on data to recognize patterns and make predictions or decisions. It has applications in various fields and is used for tasks like image recognition, natural language processing, and recommendation systems.
### What is Unsupervised vs Supervised learning difference?
- Supervised learning uses labeled data with input-output pairs for training, while unsupervised learning uses unlabeled data and focuses on finding patterns or representations within the data without explicit output labels.
### What is Deep Learning?
- Deep learning is a subset of machine learning that involves using artificial neural networks with multiple layers to learn complex patterns and representations from data. It allows computers to perform tasks like image and speech recognition, natural language processing, and decision-making at a high level of accuracy and sophistication.
### What is Neural Network (NN)?
- A neural network (NN) is a computational model inspired by the structure and functioning of the human brain. It consists of interconnected artificial neurons organized into layers. Each neuron receives input, processes it using an activation function, and produces an output. Neural networks are used in deep learning to learn and represent complex patterns and relationships in data.
### What is Convolution Neural Network (CNN)? Please give 2 advantages over NN.
- Convolutional Neural Networks (CNNs) are specialized neural networks for image processing. Their advantages over traditional NNs are:
  1. Automatic feature learning: CNNs learn relevant features from data, reducing the need for manual feature engineering.
  2. Spatial invariance: CNNs can detect patterns regardless of their location in the image, making them suitable for tasks like image classification and object detection.
### What is segmentation task in NN? Is it supervised or unsupervised?
- Segmentation task in neural networks involves dividing an image into distinct regions or segments. It can be either supervised (using labeled data) or unsupervised (without explicit labels).
### What is classification task in NN? Is it supervised or unsupervised?
- Classification tasks in neural networks involve categorizing input data into predefined classes or categories. It is a supervised learning task that requires labeled data during training to make predictions on new, unseen data.
### Compare segmentation and classification in NN.
- Segmentation divides an image into segments with pixel-level masks, while classification assigns a single label to an input. Segmentation can be supervised or unsupervised, while classification is typically supervised. Segmentation is used for object detection, while classification is used for tasks like image classification and sentiment analysis.
### What is data and dataset difference?
- In short, "data" refers to individual pieces of information, while a "dataset" is a structured collection of related data points used for analysis or machine learning tasks.
### What is the difference between supervised and unsupervised learning in terms of dataset?
- 1. Supervised Learning: Requires a labeled dataset, where each data instance is paired with its corresponding output label.
  2. Unsupervised Learning: Works with an unlabeled dataset, meaning there are no explicit output labels provided for the data instances.

# Data Preprocessing
## Extracting Masks
### What is color space ?
- Color space is a system that defines how colors are represented and organized. It provides a numerical representation of colors for various applications like image processing and graphics.
### What RGB stands for ?
- RGB stands for "Red, Green, Blue," and it is a color model where colors are represented by combining these three primary colors in different intensities.
### In Python, can we transform from one color space to another?
- Yes, in Python, we can transform from one color space to another using various libraries and functions. One popular library for working with colors and color spaces is OpenCV. OpenCV provides functions to convert images 
  between different color spaces, such as RGB to grayscale, RGB to HSV, RGB to LAB, etc.
### What is the popular library for image processing?
- OpenCV is the most popular library for image processing in Python, widely used for various tasks like image manipulation, feature detection, and computer vision due to its efficiency and versatility.

## Converting into Tensor
### Explain Computational Graph.
- A computational graph is a visual representation of mathematical operations and their connections in algorithms or models.
### What is Tensor?
- A tensor is a multi-dimensional numerical array used in machine learning and computational operations, like a versatile data container.
### What is one hot encoding?
- One-hot encoding is a method to represent categorical data as binary vectors, with a single "1" indicating the category and the rest as "0s".
### What is CUDA programming? Answer without detail.
- CUDA programming is a parallel computing platform and API developed by NVIDIA for utilizing GPUs (Graphics Processing Units) to accelerate computational tasks.

# Design Segmentation Model
### What is the difference between CNN and Fully CNN (FCNN) ?
- CNN (Convolutional Neural Network) includes convolutional and fully connected layers for various tasks. FCNN (Fully Convolutional Neural Network) eliminates fully connected layers to preserve spatial information, commonly used for tasks like image segmentation.
### What are the different layers on CNN ?
- Different layers in a CNN (Convolutional Neural Network) are:
  1. Convolutional Layers: Extract features from data.
  2. Pooling Layers: Reduce spatial dimensions while keeping essential information.
  3. Activation Layers: Introduce non-linearity using activation functions.
  4. Fully Connected Layers: Connect neurons across layers.
  5. Normalization Layers: Normalize neuron activations for stability.
  6. Dropout Layers: Randomly deactivate neurons to prevent overfitting.
### What is activation function ? Why is softmax usually used in the last layer?
- Activation Function: A mathematical operation adding non-linearity in neural networks, aiding complex pattern learning.
- Softmax is used in the last layer to convert raw scores into probabilities for easy and interpretable multiclass classification predictions.

# Train
### What is parameter in NN ?
- Parameters in a neural network are the weights and biases that the model learns during training to make accurate predictions.
### What is hyper-parameter in NN ?
- Hyperparameters in a neural network are preset settings affecting learning, like learning rate and layer count, chosen before training and influencing performance.
### We mention the dataset and we separate it into 2: training & test. In addition to them, there is a validation dataset. What is it for?
- The validation dataset is used to tune model settings (hyperparameters) and prevent overfitting during training. It helps in adjusting the model's behavior before evaluating on the final test dataset.
### What is epoch?
- An epoch is a single iteration through the entire training dataset during model training, helping the model learn patterns over multiple passes.
### What is batch?
- A batch is a group of training examples processed together in one iteration during model training. It improves computation efficiency and can be adjusted using the batch size parameter.
### What is iteration? Explain with an example: "If we have x images as data and batch size is y. Then an epoch should run z iterations."
- An iteration is one cycle of processing a batch of data through a model during training. If you have x images as data and a batch size of y, then an epoch should run z iterations, where z = x / y.
### What Is the Cost Function?
- The cost function quantifies the difference between predicted and actual values in a machine learning model. Minimizing it helps improve prediction accuracy during training.
### The process of minimizing (or maximizing) any mathematical expression is called optimization. What is/are the purpose(s) of an optimizer in NN?
- The purpose of an optimizer in a neural network is to minimize (or maximize) the cost or loss function by adjusting the model's parameters during training. Optimizers are responsible for finding the best set of parameter values that result in the lowest possible loss, which indicates better model performance. They determine how the model's parameters are updated in response to the calculated gradients of the cost function with respect to those parameters. Optimizers play a crucial role in helping neural networks learn and converge to meaningful solutions efficiently.
### What is Batch Gradient Descent & Stochastic Gradient Descent? Compare them.
- Batch Gradient Descent (BGD):

1. Computes gradients using the entire dataset.
2. Slow for large datasets.
- Stochastic Gradient Descent (SGD):

1. Computes gradients using one data point at a time.
2. Faster but noisy updates.
- Comparison:

1. BGD accurate but slow, SGD fast and noisy.
2. BGD stable, SGD escapes local minima.
3. BGD may get stuck, SGD oscillates but stabilizes.
4. Mini-batch SGD balances advantages.
- Choose based on speed and convergence requirements.

### What is Backpropogation ? What is used for ?
- Backpropagation is an algorithm to adjust neural network parameters by calculating gradients of the loss function. It's used for training the network by minimizing prediction errors through iterative parameter updates.
