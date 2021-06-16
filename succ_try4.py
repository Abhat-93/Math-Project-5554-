from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from numpy import asarray
from matplotlib import pyplot
from function import *
import numpy

p = []

for i in range(len(f)):
    # Define the arrays
    x = asarray([j/div_factor[i] for j in range(lower[i]*div_factor[i], upper[i]*div_factor[i])])
    y = asarray([f[i](j/div_factor[i]) for j in range(lower[i]*div_factor[i], upper[i]*div_factor[i])])
    print(x.min(), x.max(), y.min(), y.max())

    # Reshape them into matrices
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))

    # rescale them 
    scale_x = MinMaxScaler()
    x = scale_x.fit_transform(x)
    scale_y = MinMaxScaler()
    y = scale_y.fit_transform(y)
    print(x.min(), x.max(), y.min(), y.max())

    # Define the NN Model
    model = Sequential()
    model.add(Dense(20, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation = 'sigmoid'))

    # train the model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x, y, epochs=epoch[i], batch_size=batch_Size[i], verbose=2) # Training

    # Rescale the vectors for testing, if required
    x = asarray([j/div_factor[i] for j in range(zoom_factor[i]*lower[i]*div_factor[i], zoom_factor[i]*upper[i]*div_factor[i])])
    y = asarray([f[i](j/div_factor[i]) for j in range(zoom_factor[i]*lower[i]*div_factor[i], zoom_factor[i]*upper[i]*div_factor[i])])
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    x = scale_x.fit_transform(x)
    y = scale_y.fit_transform(y)

    # Prediction of the trained model
    yhat = model.predict(x)

    # Plot the relevant information
    x_plot = scale_x.inverse_transform(x)
    y_plot = scale_y.inverse_transform(y)
    yhat_plot = scale_y.inverse_transform(yhat)

    #print('MSE: %.3f' % mean_squared_error(y_plot, yhat_plot))

    pyplot.scatter(x_plot,y_plot, label='Actual')
    pyplot.scatter(x_plot,yhat_plot, label='Predicted')
    pyplot.title(labels[i])
    pyplot.xlabel('Input Variable (x)')
    pyplot.ylabel('Output Variable (y)')
    pyplot.legend()
    pyplot.savefig("func_plot" + str(i))
    pyplot.close()
    file = open("Close.txt", "a+")
    file.write(str(numpy.max(abs(yhat_plot - y_plot)))+ "\n")
    file.close()
    #print(numpy.max(abs(yhat_plot - y_plot)))

    #print(history.history['loss'])
    #print(history.history['accuracy'])



