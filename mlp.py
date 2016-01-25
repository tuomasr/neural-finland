import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu")
import numpy as np
import pandas as pd
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

# standardize data so that it has zero mean and unit variance
def stand_data(x):
    x_stand = (x-x.mean(axis=0))/x.std(axis=0)

    return x_stand

# normalize data to the interval [0,1]
def norm_data(x):
    x_normed = (x-x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    return x_normed

# normalize data to the interval [-1,1]
def norm_data2(x):
    x_normed = 2*(x-x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)) - 1

    return x_normed

def train_test_split(x, y, z, test_size=0.1):  
    """
    Split data to training and testing parts
    """
    n_trn = int(x.shape[0]*(1-test_size))
    
    xtrain = x[:n_trn,:]
    xtest = x[n_trn:,:]
    ytrain = y[:n_trn,:]
    ytest = y[n_trn:,:]
    ztrain = z[:n_trn,:]
    ztest = z[n_trn:,:]

    # print data shapes
    print xtrain.shape
    print xtest.shape
    print ytrain.shape
    print ytest.shape

    return (xtrain, ytrain, ztrain), (xtest, ytest, ztest)

def main():
    # read input and output data
    xall = np.loadtxt('input.csv', delimiter=",", skiprows=1)
    yall = np.loadtxt('output.csv', delimiter=",", skiprows=1)
    zall = np.loadtxt('dates.csv', delimiter=",", skiprows=1)

    yall = yall.reshape(yall.shape[0],1)
    ymax = yall.max(axis=0)
    ymin = yall.min(axis=0)
    ystd = yall.std(axis=0)
    ymean = yall.mean(axis=0)

    xall = stand_data(xall)
    yall = norm_data2(yall)

    # split the data to train and test sets
    (xtrain, ytrain, ztrain), (xtest, ytest, ztest) = train_test_split(xall, yall, zall)

    # model parameters
    nb_epoch = 40
    batch_size = 200

    # model structure
    model = Sequential()
    model.add(Dense(240, input_dim=xtrain.shape[1]))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(120))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(60))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(30))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="tanh"))

    # model fitting
    adam = Adam()
    model.compile(loss="mae", optimizer=adam)
    model.fit(xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2)

    # model evaluation
    score = model.evaluate(xtest, ytest, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # prediction
    predicted = model.predict(xtest, verbose=1)
    predicted = predicted.flatten()
    predicted = predicted.reshape(predicted.shape[0],1)
    ytest = ytest.flatten()
    ytest = ytest.reshape(ytest.shape[0],1)

    # scale back to absolute values
    predicted = (predicted+1)*(ymax-ymin)*0.5 + ymin
    ytest = (ytest+1)*(ymax-ymin)*0.5 + ymin

    # output test error
    mse = ((predicted-ytest)**2).mean(axis=0)
    mae = (np.abs(predicted-ytest)).mean(axis=0)
    print('MSE', mse)
    print('MAE', mae)

    # output results
    result = np.hstack((ztest, predicted, ytest))
    pd.DataFrame(result).to_csv("result_mlp.csv")

main()
