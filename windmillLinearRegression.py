import numpy as np
import pandas as pd

# hyper parameters
alpha = 0.001
num_epoch = 1000000

np.set_printoptions(precision=1)

raw_data = pd.read_csv("Train.csv", sep=';', header=0)

windmill_input = raw_data.values[:, (6, 10, 7, 19)]
windmill_output = raw_data.values[:, 8]

windmill_input = windmill_input / windmill_input.max(axis=0)
m = len(windmill_output)

windmill_output = windmill_output.reshape(m, 1)
new_input = np.insert(windmill_input, 0, 1, axis=1)
theta = np.matrix(np.zeros((5,1)))

def grad_descent(x1,y,theta,iter_num,alpha):
    m = len(y)
    costs=[]
    for i in range(iter_num):

        theta = theta + x1.T*(y - x1*theta)*alpha/m
        cost = 1/2/m * (x1*theta-y).T*(x1*theta-y)
        if (i%10000)==0:
            costs.append(cost)
            print('iteration = %i, cost = %.8f' %(i,cost))
    return theta, costs

new_theta,costs = grad_descent(new_input, windmill_output, theta, num_epoch, alpha)

print(new_theta)

