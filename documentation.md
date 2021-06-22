## Appendix B - Documentation

In this section of the Appendix, we attempt to highlight at a high level, some of the decisions that were taken throughout the duration of this capstone project. This section presents some of the technical details behind those decisions.

### Data Augemntations 

The team initially used a web interface provided by the albumentations library, powered by PyTorch for image transformations, that allowed us to see what the different transformations would look like. Based on that, we chose some transformations to test in our pipeline. This experiment can be found [here](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/master/notebooks/manual-albumentation.ipynb). The final model was chosen based on the transformations yielding the highest accuracy score.

To use the albumentations library:

- This [guide](https://albumentations.ai/docs/examples/pytorch_classification/) is extremely helpful for our classification task. You develop your own class that inherits from the torch dataset. To see an example of this implementation for our dataset, please check this [notebook](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/auto_exp1/notebooks/autoalbument.ipynb).
- Some transformations that are believed be helpful can be found. The following list is an example of those. 

We also tried a machine learning tool called [autoalbument](https://albumentations.ai/docs/autoalbument/) which allows for the selection of optimal transformations.

### Loss Functions

A loss function, also known as cost function or error function, is a function that maps a set of parameter values for the network onto a scalar value. The scalar value is an indication of how well the parameters are completing their assigned tasks. In optimization problems, we seek to minimize the loss function. 

The following functions were considered for the choice of the loss function used in the convolutional neural network model:

1. BCEWithLogitsLoss
2. NLLLoss
3. Sum of Log Loss
4. Hinge
5. The mean absolute error 

- BCEWithLogitsLoss: Cross-Entropy loss [nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss) combines a Sigmoid layer and the BCELoss in one single class. This loss function in often used in classification problems as a measure of reconstruction error. It uses log loss, which for logistic regression is a special case for cross-entropy loss for multi-class classification.

Alternative Loss Functions Considered

- NLLLoss: Negative Log-Likelihood Loss
This function expands the binary cross-entropy loss. It requires an additional logSoftMax layer in the last layer of the network. As it outputs probablities, they must sum to 1.
The function also has various reduction of output options such as the weighted mean of output or sum.

- Sum of Log Loss: This function allows for the gradient to update with more incentive

- Hinge: This function, nn.HingeEmbeddingLoss, is used more for measuring similarities. It tries to maximize the margin between decision boundary and data points. The main advantage is that the function penalizes incorrect predictions a lot but also correct ones that aren't confident (less). That is, confident correct predictions are not penalized at all.

- L1 MAE: The mean absolute error

### Structure of CNN Model

Based on our Liturature Review, the following transfer learning architectures were considered:

1. DenseNet
2. Inception
3. VGG
4. ResNet

The model variants that were considered are documented below, along with some notes thought to be relevant.

- VGG16: It makes the improvement over AlexNet by replacing large kernel-sized filters (**11** and **5** in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another.
- DenseNet121: DenseNet121 proved to be the best performing model given the Lipohypertrophy data. To see this exploration, click [here](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/master/notebooks/model_decision_making.ipynb). 
- ResNet50
- InceptionV3

#### Batch Normalization
We ensured all models have batch normalization implemented. This will standardize the inputs of a network improving the overall efficiency by reducing the number of epochs required to train for a given model.

#### Dropout Layer
The implementation of dropout layers within the model's architecture to reduce the generalizable error were considered. This is due to the fact that a dropout layer with random probability equal to the dropout rate, drop nodes from a network between layers. However, the dropout layers experiment with varying dropout rates did not show any success for our dataset. To see this exploration, click [here](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/master/notebooks/densemodels-ax-dropout-layers.ipynb). 

#### Recall Optimization
To reduce the generalization error, we considered varying the pos_weight argument in the loss function. Increasing the positive weights to more than 1 would mean heavier penalization (loss) on positives, in an attempt to reduce false negatives (a true positive where the model predicts is negative) and improve recall. To see this exploration, which is also used to combat the class imbalance issue, click [here](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/master/notebooks/pos-weight-exploration.ipynb).


### Hyperparameter Optimization - Bayesian Optimization with Ax

The general idea with the Bayesian method is that not every possible parameterization in the defined parameter space is explored. The process tunes parameters in a few iterations by building a surrogate model. A surrogate model is a probablistic model that uses an acquisition function to direct the next sample to make up configurations where an improvement over the current best parameterization is likely. Bayesian Optimization with Ax, a recently developed package for optimization by Facebook, relies on information from previous trials to propose better hyperparameters in the next evaluation. The Gaussian process is as follows:

1. Build "smooth" surrogate model using Gaussian processes (initially Gaussian with mean 0 and variance equal to the noise in data. The surrogate model updates to a Gaussian model with mean equal to the estimated mean and updated variance based on the previous trial. 
2. use the surrogate model to **predict** the next parameterization (estimated mean), out of the remaining ones and quantify uncertainty (updated estimated variance)
3. combine predictions and estimates to derive an acquisition function and then optimize it to find the best configuration.
4. Observe outcomes

Fit new surrogate model and repeat process.

As a way to guide the surrogate model in the prediction of the next parameter configuration to explore, aquisition functions are utilizied. 
- Three common examples include:
 1. Probability of Improvement (PI).
 2. Expected Improvement (EI).
 3. Lower Confidence Bound (LCB)
 
The most common technique used in Bayesian Optimization is the second choice: Expected Improvement. As such, that was used for our experiments found [here](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/master/notebooks/densenet-optimized.ipynb).

An Ax tutorial can be found [here](https://ax.dev/versions/latest/tutorials/tune_cnn.html).

The main advantages to Bayesian Optimization are:
- its ability to find better parameterizations with fewer iterations than grid search and;
- striking a balance between exploration and exploitation where exploration refers to trying out parameterization with high uncertainty in the outcome and exploitation refers to the surrogate model predicting likely parameterizations.

In addition to our Bayesian experiments, we consulted results in the following papers regarding classical transformations applied to small:
- [A Review of Bayesian Optimization](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
- [Bayesian Hyperparameter Optimization with BoTorch, GPyTorch and Ax](https://arxiv.org/pdf/1912.05686.pdf).

