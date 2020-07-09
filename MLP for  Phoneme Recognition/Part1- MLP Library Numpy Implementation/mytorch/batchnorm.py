# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """


        #inference
        if eval:
            self.norm = (x - self.running_mean) / (np.sqrt(self.running_var+self.eps)) # all entries are (1,10)
            self.out = self.gamma * self.norm + self.beta
            return self.out

        self.x = x #(50,10)


        #print(x.shape) # (50,10)

        # self.mean = # ???
        # self.var = # ???
        # self.norm = # ???
        # self.out = # ???
        self.mean = np.mean(self.x, axis=0, keepdims=True)  # (1,10)
        self.var = np.mean(np.square(self.x - self.mean), axis=0, keepdims=True) #(1,10)
        self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)  #(50,10)-(1,10)   /   sqrt((1,10)+cons)  = (50,10)
        self.out = self.gamma*self.norm + self.beta


        # Update running batch statistics
        self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var

        return self.out

        #raise NotImplemented


    def backward(self, delta):  # (delta) is the derivative of the loss w.r.t. Batch Norm outputin a given MLP.
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """


        #for one delta : (1,10)
        dxinorm = delta * self.gamma #(50,10)
        self.dbeta = np.sum(delta,axis=0,keepdims=True)
        self.dgamma = np.sum(delta*self.norm, axis=0,keepdims=True)
        # if(self.x is None):
        #     print("PPPPPP")
        # if(self.mean is None):  # == ? bunebngyiong
        #     print("LLLLLLL")
        dvar = -(1.0/2.0) * np.sum( dxinorm * (self.x - self.mean) * ((self.var + self.eps)  ** (-3.0/2.0)) ,axis=0,keepdims=True) #(1,10)

        dmiubfirst = - np.sum(dxinorm * ((self.var+self.eps) ** (-1.0/2.0)) ,axis=0,keepdims=True)
        dmiubsecond = -2.0 * dvar * np.mean( self.x - self.mean ,axis=0,keepdims=True)

        dmiub = dmiubfirst + dmiubsecond

        dxi  = dxinorm * ((self.var + self.eps)**(-1.0/2.0))+ dvar * (2.0/delta.shape[0] * (self.x - self.mean)) + (1.0/delta.shape[0]) * dmiub



        return dxi


        # raise NotImplemented
