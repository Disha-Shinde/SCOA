import numpy as np

class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 1 else 0
 
    def predict(self, x):
        z = self.W.T.dot(x)
        #print(z)
        a = self.activation_fn(z)
        return a
 
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                #print(x)
                #print(y)
                e = d[i] - y
                #print(e)
                self.W = self.W + self.lr * e * x
                print(self.W)
                
if __name__ == '__main__':
    #---AND/OR---
    '''
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    #AND
    d = np.array([0, 0, 0, 1])
    #OR
    #d = np.array([0, 1, 1, 1])
    
    perceptron = Perceptron(input_size=2)
    '''
    #-----------
    
    #----NOT---- 
    X = np.array([[0], [1]])
    d = np.array([1, 0])    
    perceptron = Perceptron(input_size=1)
    #-----------
    
    perceptron.fit(X, d)
    print(perceptron.W)           
    
    
    
    
    
         
