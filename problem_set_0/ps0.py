import numpy as np
import matplotlib.pyplot as plt 

def main():
    '''
    Problem 10
    #scale
    axes = plt.gca()
    axes.set_xlim([-4, 4])
    axes.set_ylim([-4, 4])

    mean = [0, 0]
    cov = [[1, -0.5], [-0.5, 1]]
    x = np.random.multivariate_normal(mean, cov, 1000)
    y = np.random.multivariate_normal(mean, cov, 1000)
    points = np.vstack((x, y))
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()
    '''

    A = [[1, 0], [1, 3]]
    print(np.linalg.eigvals(A))

if __name__ == '__main__':
    main()

