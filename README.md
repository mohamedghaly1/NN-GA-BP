# Neural Network Training using Genetic Algorithm and Backpropagation
## Overview
This project focuses on building and training a neural network (NN) for a classification task, utilizing both Genetic Algorithm (GA) and Backpropagation for weight optimization. The dataset is selected from the UCI Machine Learning Repository, and various preprocessing steps are applied before training. The neural network's performance is evaluated based on training, validation, and testing accuracy.

## Project Tasks
### Task 1: Dataset Selection and Preprocessing
#### Dataset Selection:
A dataset from the UCI Machine Learning Repository is chosen, addressing a real-world classification problem.
The significance of solving this problem is highlighted, along with the rationale for using a Neural Network due to its ability to capture complex, non-linear relationships.
#### Data Preprocessing:
Visualization techniques (histograms, scatter plots, bar graphs) are used to analyze feature distributions and detect patterns or outliers.
Feature identification (numeric, categorical, text, etc.), including descriptive statistics such as mean, median, and standard deviation for numerical data.
Handling missing values by either removal or applying appropriate imputation techniques.
All dataset selection, preprocessing steps, and visual outputs are documented in detail.

### Task 2: Neural Network Training with Genetic Algorithm (GA)
A neural network with the architecture [N, 10, 1] is implemented, where:

N neurons in the input layer (determined by dataset features).
1 hidden layer with 10 neurons.
1 output neuron (for binary classification).
#### Training Process
##### Data Splitting:

Training set (T = 70%)
Validation set (V = 20%)
Testing set (S = 10%)
##### Batch Processing:

Data is divided into batches of 100 samples.
The model is trained for at least 10 epochs.
##### Genetic Algorithm for Weight Optimization:

Individual Representation: Each individual in the GA population represents a set of network weights and biases.
Fitness Function: The fitness of an individual is evaluated based on validation performance.
Evolution Process:
Selection → Fitter individuals are selected for reproduction.
Crossover → Parent weights are combined to create offspring.
Mutation → Small random modifications (mutations) ensure diversity.
##### Validation and Testing:

A custom testing function is implemented to calculate error after each forward pass.
The GA process is executed after every batch, validation is performed after each epoch, and final testing is conducted at the end of training.
Training, validation, and testing accuracies are reported with tables, graphs, and charts.
### Task 3: Neural Network Training with Backpropagation
A second version of the neural network is implemented, replacing the GA-based weight optimization with Backpropagation.

#### Backpropagation Algorithm:

The network is trained using gradient descent, updating weights by propagating errors backward.
Training function:
python
Copy
Edit
Weights_Matrix = train_[Teamname](Labeled_Input_Matrix, Weights_Matrix, BatchSize)
The function consists of:
Forward pass → Computes activations for each layer.
Backward pass → Calculates gradients and updates weights.
#### Same Training Pipeline as GA-Based Approach:

Data is split and processed identically.
The error function and evaluation process are the same.
Training performance is compared with GA-based optimization.
## Expected Outcomes
A fully functional neural network trained using both Genetic Algorithm and Backpropagation.
Comparison of performance between the two training methods in terms of accuracy, error rate, and computational efficiency.
Detailed analysis through graphs, charts, and tables showcasing training progress and model performance.
