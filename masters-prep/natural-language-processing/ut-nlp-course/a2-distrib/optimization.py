# optimization.py

import argparse
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

### 
# IMPLEMENT ME! REPLACE WITH YOUR ANSWER TO PART 1B
OPTIMAL_STEP_SIZE = 0.11 
###

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='optimization.py')
    parser.add_argument('--func', type=str, default='QUAD', help='function to optimize (QUAD or NN)')
    parser.add_argument('--lr', type=float, default=1., help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    return args


def quadratic(x1, x2):
    """
    Quadratic function of two variables
    :param x1: first coordinate
    :param x2: second coordinate
    :return:
    """
    return (x1 - 1) ** 2 + 8 * (x2 - 1) ** 2


def quadratic_grad(x1, x2):
    """
    Should return a numpy array containing the gradient of the quadratic function defined above evaluated at the point
    :param x1: first coordinate
    :param x2: second coordinate
    :return: a two-dimensional numpy array containing the gradient
    """
    return np.array([2 * (x1 - 1), 16 * (x2 - 1)])
    #raise Exception("Implement me!")


def sgd_test_quadratic(args):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = quadratic(X, Y)
    plt.figure()

    # Track the points visited here
    points_history = []
    curr_point = np.array([0, 0])
    for iter in range(0, args.epochs):
        grad = quadratic_grad(curr_point[0], curr_point[1])
        if len(grad) != 2:
            raise Exception("Gradient must be a two-dimensional array (vector containing [df/dx1, df/dx2])")
        next_point = curr_point - args.lr * grad
        points_history.append(curr_point)
        
        ## BEFORE
        #print("Point after epoch %i: %s" % (iter, repr(next_point)))
        ##
        
        ## AFTER
        formatted_next_point = f"[{', '.join(f'{n:07.5f}' for n in next_point)}]"
        print("Point after epoch %02.i: %s %s" % (iter, formatted_next_point, ([1, 1] - next_point) < 0.1))
        ##
        
        curr_point = next_point
    points_history.append(curr_point)
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.plot([p[0] for p in points_history], [p[1] for p in points_history], color='k', linestyle='-', linewidth=1, marker=".")
    plt.title('SGD on quadratic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    exit()

# how do I block select in vs code? 
    #lr = 0.03   # 37
    #lr = 0.04   # 27
    #lr = 0.05   # 21
    #lr = 0.06   # 18
    #lr = 0.063  # 17
    #lr = 0.07   # 15
    #lr = 0.075  # 14
    #lr = 0.079  # 13
    #lr = 0.078  # 13
    #lr = 0.0795 # 13
    #lr = 0.085  # 12
    #lr = 0.08   # 13
    #lr = 0.083  # 12
    #lr = 0.086  # 12
    #lr = 0.087  # 12
    #lr = 0.09   # 11
    #lr = 0.1    # 10
    #lr = 0.105  # 9
    #lr = 0.11   # 9
    #lr = 0.1115 # 9
    #lr = 0.112  # 9
    #lr = 0.1125 #10
    #lr = 0.115  # 12
    #lr = 0.12   # 26
    #lr = 0.13   # XXX
    #lr = 0.14   # XXX
    #lr = 0.15   # XXX
    #lr = 0.2    # XXX

if __name__ == '__main__':
    args = _parse_args()
    sgd_test_quadratic(args)
