"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum


        #added
        self.output = None
        self.loss = 0.0
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        hiddens.insert(0,input_size)
        hiddens.append(output_size)
        self.linear_layers = [Linear(hiddens[i],hiddens[i+1],weight_init_fn,bias_init_fn)
                              for  i in range(len(hiddens)-1) ]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(hiddens[i+1]) for i in range(num_bn_layers)]

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        #raise NotImplemented

        xCopy = x.copy()

        out = x.copy()  # in case the net in empty

        batchleft = self.num_bn_layers

        for i in range(len(self.linear_layers)):

            out = self.linear_layers[i].forward(xCopy)

            if batchleft:

                out = self.bn_layers[-batchleft].forward(out,not self.train_mode)

                batchleft = batchleft - 1

            out = self.activations[i].forward(out)
            xCopy = out

        self.output = out

        return out


    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        #raise NotImplemented
        for i in range(len(self.linear_layers)):
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)

        #for batch norm
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].dgamma.fill(0.0)
                self.bn_layers[i].dbeta.fill(0.0)



    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            #pass
            self.linear_layers[i].momentum_W = self.momentum * self.linear_layers[i].momentum_W - self.lr * self.linear_layers[i].dW
            self.linear_layers[i].W = self.linear_layers[i].W + self.linear_layers[i].momentum_W

            self.linear_layers[i].momentum_b = self.momentum * self.linear_layers[i].momentum_b - self.lr * self.linear_layers[i].db
            self.linear_layers[i].b = self.linear_layers[i].b + self.linear_layers[i].momentum_b



        # Do the same for batchnorm layers

        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
                self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta

        #raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        #raise NotImplemented
        self.loss = self.total_loss(labels)

        #delta = self.criterion.derivative() * self.activations[-1].derivative()  # element wise
        dsoftmax = self.criterion.derivative()
        delta = dsoftmax * self.activations[-1].derivative()

        batchLeftIndex = -1

        if self.bn:
            batchLeftIndex = self.num_bn_layers -1


        for i in range(len(self.linear_layers)-1,-1,-1):    # i from len-1 to 0 with step 1
            if i == len(self.linear_layers)-1:
                if i == batchLeftIndex:
                    delta = self.bn_layers[batchLeftIndex].backward(delta)
                    batchLeftIndex = batchLeftIndex - 1
                delta = self.linear_layers[i].backward(delta)

            else:
                delta = delta * self.activations[i].derivative()

                if i == batchLeftIndex:
                    delta = self.bn_layers[batchLeftIndex].backward(delta)
                    batchLeftIndex = batchLeftIndex - 1

                delta = self.linear_layers[i].backward(delta)


    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def weight_init(x, y):
    return np.random.randn(x, y)


def bias_init(x):
    return np.zeros((1, x))

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val


    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...



    for e in range(nepochs):
        print("On epcoh: ", e+1)
        # Per epoch setup ...
        trainBatchLoss = 0.0
        trainBatchError = 0.0
        valBatchLoss = 0.0
        valBatchError = 0.0

        np.random.shuffle(idxs)

        for b in range(0, len(trainx), batch_size):


            #pass  # Remove this line when you start implementing this
            # Train ...
            mlp.zero_grads()
            mlp.forward(trainx[idxs[b : b + batch_size]])
            mlp.backward(trainy[idxs[b : b + batch_size]])

            trainBatchLoss = trainBatchLoss + mlp.total_loss(trainy[idxs[b : b + batch_size]]) / batch_size

            trainBatchError = trainBatchError + mlp.error(trainy[idxs[b : b + batch_size]]) / batch_size

            mlp.step()

        training_losses[e] = trainBatchLoss / ((len(trainx)) / batch_size)
        training_errors[e] = trainBatchError / ((len(trainx)) / batch_size)

        for b in range(0, len(valx), batch_size):

            #pass  # Remove this line when you start implementing this
            # Val ...

            mlp.forward(valx[b : b + batch_size])

            valBatchLoss =valBatchLoss +  mlp.total_loss(valy[b : b + batch_size]) / batch_size
            valBatchError = valBatchError + mlp.error(valy[b : b + batch_size]) / batch_size

        validation_losses[e] = valBatchLoss / ((len(valx)) / batch_size)
        validation_errors[e] = valBatchError / ((len(valx)) / batch_size)

        # Accumulate data...

    # Cleanup ...

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

    #raise NotImplemented
