# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.x = None
        self.out_size = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x = x
        batch_size = x.shape[0]
        input_size = x.shape[2]

        out_size = (input_size - self.kernel_size + self.stride) // self.stride
        self.out_size = out_size
        out = np.zeros([batch_size,self.out_channel,self.out_size])


        for i in range(batch_size):
            for j in range(self.out_channel):
                for k in range(self.out_size):
                    out[i,j,k] = np.multiply( x[i,:,k*self.stride : k*self.stride+self.kernel_size], self.W[j,:,:] ).sum()
                out[i,j]  = out[i,j] + self.b[j]
        #raise NotImplemented
        return out



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """

        dx = np.zeros(self.x.shape)
        self.db = np.sum(delta, axis=(0, 2))
        batch_size = delta.shape[0]
        output_size = delta.shape[2]

        for batch in range(batch_size):
            for m in range(self.kernel_size):
                for n in range(self.in_channel):
                    for p in range(self.out_channel):
                        for q in range(output_size):
                            self.dW[p,n,m] += self.x[batch,n,m+self.stride*q]*delta[batch,p,q]

        for batch in range(batch_size):
            for m in range(self.kernel_size):
                for n in range(self.in_channel):
                    for p in range(self.out_channel):
                        for q in range(self.out_size):
                            dx[batch,n,self.stride*q+m] += delta[batch,p,q] * self.W[p,n,m]

        return dx

        #raise NotImplemented



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        xcopy = x.copy()
        return xcopy.reshape(self.b,self.c*self.w)
        #raise NotImplemented

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        deltacopy = delta.copy()
        dx = deltacopy.reshape(self.b, self.c, self.w)
        return dx
        #raise NotImplemented
