# version 1.0

from typing import List
import numpy as np

from operations import *

class NeuralNetwork():
    '''
    A class for a fully connected feedforward neural network (multilayer perceptron).
    :attr n_layers: Number of layers in the network
    :attr activations: A list of Activation objects corresponding to each layer's activation function
    :attr loss: A Loss object corresponding to the loss function used to train the network
    :attr learning_rate: The learning rate
    :attr W: A list of weight matrices. The first row corresponds to the biases.
    '''

    def __init__(self, n_features: int, layer_sizes: List[int], activations: List[Activation], loss: Loss,
                 learning_rate: float=0.01, W_init: List[np.ndarray]=None):
        '''
        Initializes a NeuralNetwork object
        :param n_features: Number of features in each training examples
        :param layer_sizes: A list indicating the number of neurons in each layer
        :param activations: A list of Activation objects corresponding to each layer's activation function
        :param loss: A Loss object corresponding to the loss function used to train the network
        :param learning_rate: The learning rate
        :param W_init: If not None, the network will be initialized with this list of weight matrices
        '''

        sizes = [n_features] + layer_sizes
        if W_init:
            assert all([W_init[i].shape == (sizes[i] + 1, sizes[i+1]) for i in range(len(layer_sizes))]), \
                "Specified sizes for layers do not match sizes of layers in W_init"
        assert len(activations) == len(layer_sizes), \
            "Number of sizes for layers provided does not equal the number of activations provided"

        self.n_layers = len(layer_sizes)
        self.activations = activations
        self.loss = loss
        self.learning_rate = learning_rate
        self.W = []
        for i in range(self.n_layers):
            if W_init:
                self.W.append(W_init[i])
            else:
                rand_weights = np.random.randn(sizes[i], sizes[i+1]) / np.sqrt(sizes[i])
                biases = np.zeros((1, sizes[i+1]))
                self.W.append(np.concatenate([biases, rand_weights], axis=0))
   
    #  remember to pass in layer_weight copy
    def forward_one(self, layer_weight:np.ndarray, example:np.ndarray, layer:int) -> (np.ndarray, np.ndarray):
        for i in range(example.shape[0]):
            val = example[i]
            layer_weight[i+1] *= val
        layer_weight = layer_weight.transpose()
          
        a_vals = layer_weight.sum(axis=1)
        z_vals = self.activations[layer].value(a_vals.copy())
        return a_vals, z_vals  


    def forward_pass(self, X) -> (List[np.ndarray], List[np.ndarray]):
        '''
        Executes the forward pass of the network on a dataset of n examples with f features. Inputs are fed into the
        first layer. Each layer computes Z_i = g(A_i) = g(Z_{i-1}W[i]).
        :param X: The training set, with size (n, f)
        :return A_vals: a list of a-values for each example in the dataset. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
                Z_vals: a list of z-values for each example in the dataset. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
        '''

        #####################################
        # YOUR CODE HERE
        #####################################
        final_a_vals = []
        final_z_vals = []
        newX = X
        for i in range(len(self.W)):
            current_a_vals = []
            current_z_vals = []
            weight = self.W[i]
            for example in newX:
                newa_vals, newz_vals = self.forward_one(weight.copy(), example, i)
                current_a_vals.append(newa_vals)
                current_z_vals.append(newz_vals)
            
            
            final_a_vals.append(np.array(current_a_vals,ndmin = 2))
            final_z_vals.append(np.array(current_z_vals,ndmin = 2))
            newX = current_z_vals
        return final_a_vals, final_z_vals

    def backward_one(self, a_vals, last_layer:np.array, layer:int):
        weight = self.W[layer+1][1:]
        newVal = last_layer * weight
        newVal = newVal.sum(axis=1)
        for i in range(len(a_vals)):
            a_val = a_vals[i]
            newVal[i] *= self.activations[layer].derivative(a_val)
            
        return newVal
        
        
    def backward_pass(self, A_vals, dLdyhat) -> List[np.ndarray]:
        '''
        Executes the backward pass of the network on a dataset of n examples with f features. The delta values are
        computed from the end of the network to the front.
        :param A_vals: a list of a-values for each example in the dataset. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :param dLdyhat: The derivative of the loss with respect to the predictions (y_hat), with shape (n, layer_sizes[-1])
        :return deltas: A list of delta values for each layer. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
        '''

        #####################################
        # YOUR CODE HERE
        #####################################
        # BASE CASE
        result = []
        lastLayer = []
        for i in range(len(dLdyhat)):
            dyHat = dLdyhat[i]
            a = A_vals[-1][i]
            da = self.activations[-1].derivative(a)
            val = dyHat * da
            lastLayer.append(val)
        lastLayer = np.array(lastLayer)
        result.append(lastLayer)
        for i in range(len(self.W)-2, -1, -1):
            current_delta = []
           
            for k in range(len(A_vals[i])):
                a_val = A_vals[i][k]
                new_delta = self.backward_one(a_val, lastLayer[k], i)
                current_delta.append(new_delta)
            result.append(np.array(current_delta,ndmin = 2))
            lastLayer = current_delta
        result.reverse()
        return result

    def update_weights(self, X, Z_vals, deltas) -> List[np.ndarray]:
        '''
        Having computed the delta values from the backward pass, update each weight with the sum over the training
        examples of the gradient of the loss with respect to the weight.
        :param X: The training set, with size (n, f)
        :param Z_vals: a list of z-values for each example in the dataset. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :param deltas: A list of delta values for each layer. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :return W: The newly updated weights (i.e. self.W)
        '''

        #####################################
        # YOUR CODE HERE
        #####################################
        numofExample = X.shape[0]
        # i - layer index
        
        for i in range(len(deltas)):
            
            holder = np.zeros(self.W[i].shape)
            layer = deltas[i]
            # n - example index
            # for n in range(len(layer)):
            #     example = layer[n]
            for j in range(holder.shape[0]):
                for k in range(holder.shape[1]):      
                    if (i == 0):
                        if (j == 0):
                            holder[j][k] += (layer.transpose()[k] * 1).sum()
                        else:  
                            holder[j][k] += (layer.transpose()[k] * X.transpose()[j-1]).sum()
                    else:
                        if (j == 0):
                            holder[j][k] += (layer.transpose()[k] * 1).sum()
                        else:
                            holder[j][k] += (layer.transpose()[k] * Z_vals[i-1].transpose()[j-1]).sum()
                  
            self.W[i] -= holder * self.learning_rate
        return self.W

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> (List[np.ndarray], List[float]):
        '''
        Trains the neural network model on a labelled dataset.
        :param X: The training set, with size (n, f)
        :param y: The targets for each example, with size (n, 1)
        :param epochs: The number of epochs to train the model
        :return W: The trained weights
                epoch_losses: A list of the training losses in each epoch
        '''
        
        
        
        epoch_losses = []
        for epoch in range(epochs):
            A_vals, Z_vals = self.forward_pass(X)   # Execute forward pass
            y_hat = Z_vals[-1]                      # Get predictions
            L = self.loss.value(y_hat, y)           # Compute the loss
            print("Epoch {}/{}: Loss={}".format(epoch, epochs, L))
            epoch_losses.append(L)                  # Keep track of the loss for each epoch

            dLdyhat = self.loss.derivative(y_hat, y)         # Calculate derivative of the loss with respect to output
            deltas = self.backward_pass(A_vals, dLdyhat)     # Execute the backward pass to compute the deltas
            self.W = self.update_weights(X, Z_vals, deltas)  # Calculate the gradients and update the weights

        return self.W, epoch_losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric) -> float:
        '''
        Evaluates the model on a labelled dataset
        :param X: The examples to evaluate, with size (n, f)
        :param y: The targets for each example, with size (n, 1)
        :param metric: A function corresponding to the performance metric of choice (e.g. accuracy)
        :return: The value of the performance metric on this dataset
        '''

        A_vals, Z_vals = self.forward_pass(X)       # Make predictions for these examples
        y_hat = Z_vals[-1]
        metric_value = metric(y_hat, y)     # Compute the value of the performance metric for the predictions
        return metric_value