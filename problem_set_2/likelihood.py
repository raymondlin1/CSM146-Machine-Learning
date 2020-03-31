import numpy as np 
import matplotlib.pyplot as plt 

def likelihood(num_success, total_num, probability):
    res = 1.
    for i in range(num_success):
        res *= probability
    
    num_failures = total_num - num_success
    probability = (1 - probability)
    for i in range(num_failures):
        res *= probability
    
    return res

def main():
    thetas = np.linspace(0., 1.0, num=101)
    likelihoods = []
    num_success = 5
    total_num = 10
    for val in thetas:
        likelihoods.append(likelihood(num_success, total_num, val))
    
    maximum = np.max(likelihoods)
    index = np.where(likelihoods == maximum)
    point1 = [thetas[index]] 
    point2 = [0]
    plt.scatter(point1, point2)
    plt.plot(thetas, likelihoods)
    plt.ylim(0., 0.002)
    plt.title("Likelihoods with Varying Theta Values")
    plt.xlabel("Theta")
    plt.ylabel("Likelihood")
    plt.show()
    
if __name__ == '__main__':
    main()