# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        self.loss = []
        #print("SHAPE IS: ", self.logits.shape)
        for i in range(x.shape[0]):
            l = 0.0
            c = max(x[i])

            for j in range(x.shape[1]):

                logsoftmax = x[i][j] - np.log( sum( np.exp( [e-c for e in x[i]] ) ) ) - c
                l = l + y[i][j] * logsoftmax

            l = -l
            self.loss.append(l)

        return np.array(self.loss)
        #raise NotImplemented

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        batchData = []

        for i in range (self.logits.shape[0]):
            #c = max(self.logits[i])

            sumOneLine = sum(np.exp(self.logits[i]))

            oneData = []
            for j in range ( self.logits.shape[1]):
                #logsoftmax = self.logits[i][j] - np.log(sum(np.exp([x - c for x in self.logits[i]]))) - c

                softmax = np.exp(self.logits[i][j]) / sumOneLine

                oneData.append(softmax)
            batchData.append(oneData)

        res = np.array(batchData)-self.labels

        #print("res shape:" , res.shape)

        return res

        #raise NotImplemented
