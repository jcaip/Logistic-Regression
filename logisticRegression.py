import numpy as np
import numpy.linalg as npla
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib
importlib.import_module('mpl_toolkits.mplot3d').__path__

class linearRegression:
    X, y, theta = 0, 0,0

    def sigmoid(z):
        #sigmoid activation function

    def sigmoidGradient(z):
        #sigmoid activation function gradient for gradient checking

    def costFunction(self):
        #evaluates the costFunction of the object right now

    def gradientDescent(self, alpha, num_iter):
        #runs gradient descent on the current object to minimize the cost function

    def plot(self):
        #plot the regression model and the dataset

    def __init__(self, dataset_name):
        #initializes the class

    @classmethod
    def trainGD(cls, dataset_name, alpha, num_iter=400): # initializes the lin reg to run with grad descent
        lin_reg = cls(dataset_name)
        lin_reg.featureNormalize()
        lin_reg.gradientDescent(alpha, num_iter)
        return lin_reg

lin_reg = linearRegression.trainGD("ex1.txt", 0.01)
lin_reg.plot()
