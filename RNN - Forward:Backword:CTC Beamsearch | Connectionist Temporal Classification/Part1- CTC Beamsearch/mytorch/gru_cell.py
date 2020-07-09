import numpy as np
from activation import *

class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here


    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        self.z = self.z_act( np.dot(h,self.Wzh.T) + np.dot(x,self.Wzx.T) )

        self.r = self.r_act( np.dot(h,self.Wrh.T) + np.dot(x,self.Wrx.T) )

        self.h_tilda = self.h_act( np.dot(np.multiply(self.r,self.hidden),self.Wh.T) + np.dot(x,self.Wx.T) )

        h_t = np.multiply(1 - self.z, self.hidden) + np.multiply(self.z, self.h_tilda)

        assert self.x.shape == (self.d, )
        assert self.hidden.shape == (self.h, )

        assert self.r.shape == (self.h, )
        assert self.z.shape == (self.h, )
        assert self.h_tilda.shape == (self.h, )
        assert h_t.shape == (self.h, )

        return h_t
        #raise NotImplementedError


    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # 1) Reshape everything you saved in the forward pass.
        # 2) Compute all of the derivatives
        # 3) Know that the autograders the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        dzt = np.multiply(self.z_act.derivative(),np.multiply(self.h_tilda-self.hidden,delta))

        dht_tilda = np.multiply(self.h_act.derivative(),np.multiply(self.z,delta))

        self.dWx = np.outer(dht_tilda.T,self.x)
        self.dWh = np.outer(dht_tilda.T,np.multiply(self.r,self.hidden))

        drt = np.multiply(np.multiply( np.dot(dht_tilda,self.Wh),self.hidden),self.r_act.derivative())

        dh = np.multiply(1 - self.z, delta)
        dx = np.dot(dht_tilda, self.Wx)

        dx += np.dot(dzt,self.Wzx)
        dh += np.dot(dzt,self.Wzh) + np.multiply(np.dot(dht_tilda, self.Wh), self.r)

        self.dWzh = np.outer(dzt.T,self.hidden)
        self.dWzx = np.outer(dzt.T,self.x)

        dx += np.dot(drt,self.Wrx)
        dh += np.dot(drt,self.Wrh)

        self.dWrh = np.outer(drt.T,self.hidden)
        self.dWrx = np.outer(drt.T,self.x)

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        #raise NotImplementedError
