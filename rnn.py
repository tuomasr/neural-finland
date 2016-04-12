import numpy as np
from helper import stand_data, norm_data, norm_data_reverse
from keras.models import Sequential  
from keras.layers.core import TimeDistributedDense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM


def to_seq(x, n_prev = 48, n_pred = 48):
    xseq = x[:n_prev,:]

    for i in range(1,(x.shape[0]-n_prev)/n_pred):
        xseq = np.dstack((xseq,x[i*n_pred:n_prev+i*n_pred,:]))

    xseq = np.rollaxis(xseq, 2, 0)

    return xseq

def train_test_split(X, y, t, test_size=0.1):  
    """
    Split data to training and testing parts
    """
    X_seq = to_seq(X)
    y_seq = to_seq(y)
    t_seq = to_seq(t)
    n_trn = int(X_seq.shape[0]*(1-test_size))
    
    X_train = X_seq[:n_trn,:,:]
    X_test = X_seq[n_trn:,:,:]
    y_train = y_seq[:n_trn,:,:]
    y_test = y_seq[n_trn:,:,:]
    t_train = t_seq[:n_trn,:,:]
    t_test = t_seq[n_trn:,:,:]

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
    nb_epoch = 1000
    batch_size = 200

    # model structure
    model = Sequential()
    model.add(LSTM(50, input_dim=X_train.shape[2], return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(10, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(TimeDistributedDense(1))

    # model fitting
    model.compile(loss="mae", optimizer="rmsprop")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2)

    # model evaluation
    score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # prediction
    prediction = model.predict(X_test, verbose=1)

    # flatten prediction, y_test, and t_test from multidimensional arrays to N x 1 vectors
    prediction = prediction.flatten()
    prediction = prediction.reshape(prediction.shape[0],1)
    y_test = y_test.flatten()
    y_test = y_test.reshape(y_test.shape[0],1)
    t_test = t_test.reshape(t_test.shape[0]*t_test.shape[1],2)

    # # scale back to absolute values
    prediction = norm_data_reverse(prediction, -1, 1, y_min, y_max)
    y_test = norm_data_reverse(y_test, -1, 1, y_min, y_max)

    # write the results to a file
    result = np.concatenate((t_test, prediction, y_test), axis=1)
    np.savetxt('result_rnn.csv', result, delimiter=',', header='date,hour,prediction,real', fmt='%1.3f', comments='')

main()
