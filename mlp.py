import numpy as np
from helper import stand_data, norm_data, norm_data_reverse
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization

def train_test_split(X, y, t, test_size=0.1):  
    """
    Split data to training and testing parts
    """

    n_trn = int(X.shape[0]*(1-test_size)) # leave the last test_size percent of the data for testing
    
    X_train = X[:n_trn,:]
    X_test = X[n_trn:,:]
    y_train = y[:n_trn]
    y_test = y[n_trn:]
    t_train = t[:n_trn,:] # timestamp
    t_test = t[n_trn:,:]

    # report the dimensions of the train and test datasets
    print 'X_train dimensions:', X_train.shape
    print 'X_test dimensions:', X_test.shape
    print 'y_train dimensions:', y_train.shape
    print 'y_test dimensions:', y_test.shape
    print 't_train dimensions:', t_train.shape
    print 't_test dimensions:', t_test.shape

    return (X_train, y_train, t_train), (X_test, y_test, t_test)

def main():
    # read input and output data
    X = np.loadtxt('input.csv', delimiter=",", skiprows=1)
    y = np.loadtxt('output.csv', delimiter=",", skiprows=1)

    # the first two columns of the input data are dates and hours
    t = X[:,0:2]
    X = X[:,2:] # ignore dates and hours in training

    # by default y has one dimension (of size N) but it becomes easier to work with if it has dimensions Nx1
    y = y.reshape(y.shape[0],1)

    # save output statistics for scaling back to absolute values
    y_max = y.max(axis=0)
    y_min = y.min(axis=0)
    y_std = y.std(axis=0)
    y_mean = y.mean(axis=0)

    # standardize and normalize data
    X = stand_data(X)
    y = norm_data(y, -1, 1)

    # split the data to train and test sets
    (X_train, y_train, t_train), (X_test, y_test, t_test) = train_test_split(X, y, t)

    # model parameters
    nb_epoch = 100
    batch_size = 200

    # model structure
    model = Sequential()
    model.add(Dense(240, input_dim=X_train.shape[1]))
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

    model.add(Dense(1, activation="tanh")) # output was scaled to [-1, 1]

    # model fitting
    model.compile(loss="mae", optimizer='rmsprop') # mae works better in this case
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2)

    # model evaluation
    score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # prediction
    prediction = model.predict(X_test, verbose=1)

    # # scale back to absolute values
    prediction = norm_data_reverse(prediction, -1, 1, y_min, y_max)
    y_test = norm_data_reverse(y_test, -1, 1, y_min, y_max)

    # write the results to a file
    result = np.concatenate((t_test, prediction, y_test), axis=1)
    np.savetxt('result_mlp.csv', result, delimiter=',', header='date,hour,prediction,real', fmt='%1.3f', comments='')

main()
