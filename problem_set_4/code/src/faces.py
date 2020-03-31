"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *
import time

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    return np.random.choice(np.array(points), size=(1, k), replace=False)
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2f: implement
    k_clusters = ClusterSet()

    labels = []
    d = {}
    for p in points:
        d[p.label] = 1
    
    for key in d.keys():
        labels.append(key)

    for l in labels:
        temp = []
        for p in points:
            if l == p.label:
                temp.append(p)
        clust = Cluster(temp)
        k_clusters.members.append(clust)
        temp = []

    return k_clusters.medoids()
    ### ========== TODO : END ========== ###


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO : START ========== ###
    # part 2c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).

    return kAverages(points, k, ClusterSet.centroids, init, plot)
    ### ========== TODO : END ========== ###

def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement
    return kAverages(points, k, ClusterSet.medoids, init, plot)
    ### ========== TODO : END ========== ###

# helper functions
def equal_centers(c1, c2):
    if c1 == [] or c2 == []:
        return False

    if len(c1) != len(c2):
        return False
    
    for i in range(len(c1)):
        if type(c1[i].attrs) is not np.ndarray:
            attributes1 = np.asarray(c1[i].attrs)[0]
            attributes2 = np.asarray(c2[i].attrs)[0]
        else:
            attributes1 = c1[i].attrs
            attributes2 = c2[i].attrs
        
        attributes1 = attributes1.flatten()
        attributes2 = attributes2.flatten()
        
        for j in range(len(attributes1)):
            if attributes1[j] != attributes2[j]:
                return False
    
    return True

def kAverages(points, k, average, init='random', plot=True):
    # create a new clusterset object 
    k_clusters = ClusterSet()

    # initialize dictionary of point names to tuple of attributes and labels
    d = {}

    # initialize centers (medoids or centroids) to random data points
    old_centers = []
    if init == "random":
        centers = random_init(points, k)[0].tolist()
    else:
        centers = cheat_init(points)

    index = 0
    while not equal_centers(old_centers, centers):
        index += 1
        # for each datapoint, compute which center it is closest to
        # in the dictionary, update the point's label
        for p in points:
            closest_cent = centers[0]
            least_dist = p.distance(closest_cent)
            for cent in centers:
                dist = p.distance(cent)
                if dist < least_dist:
                    least_dist = dist
                    closest_cent = cent
            
            d[p.name] = (p.attrs, centers.index(closest_cent) + 1)

        # create a new clusterset object
        k_clusters = ClusterSet()

        # add clusters to new clusterset object
        for i in range(1, k + 1):
            cluster_points = []
            for p in points:
                if d[p.name][1] == i:
                    cluster_points.append(p)

            cluster = Cluster(cluster_points)
            k_clusters.members.append(cluster)
            cluster_points = []
            
        # compute new centers
        old_centers = centers
        if average == ClusterSet.centroids:
            centers = k_clusters.centroids()
            title = "K-Means Iteration {}".format(index)
        else:
            centers = k_clusters.medoids()
            title = "K-Medoids Iteration {}".format(index)

        '''for i in range(len(old_centers)):
            print(old_centers[i])
            print(centers[i])
        print('\n')'''
        if plot:
            plot_clusters(k_clusters, title, average)

    return k_clusters

######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    # 1a: show images 
    X, y = get_lfw_data()
    #show_image(X[0])
    #show_image(X[100])
    #show_image(X[1000])

    mu = X.mean(0)
    #show_image(mu)

    # 1b: eigenfaces
    U = PCA(X)
    #show_image(vec_to_image(U[0][:, 0]))
    #show_image(vec_to_image(U[0][:, 1]))
    #show_image(vec_to_image(U[0][:, 2]))
    #show_image(vec_to_image(U[0][:, 3]))
    #show_image(vec_to_image(U[0][:, 4]))
    #show_image(vec_to_image(U[0][:, 5]))
    #show_image(vec_to_image(U[0][:, 6]))
    #show_image(vec_to_image(U[0][:, 7]))
    #show_image(vec_to_image(U[0][:, 8]))
    #show_image(vec_to_image(U[0][:, 9]))
    #show_image(vec_to_image(U[0][:, 10]))
    #show_image(vec_to_image(U[0][:, 11]))

    # 1c: reconstruct from PCA
    '''li = [1, 10, 50, 100, 500, 1288]
    for l in li:
        Z, Ul = apply_PCA_from_Eig(X, U[0], l, mu)
        for i in range(0, 12):
            im_name = "l{}_im{}".format(l, (i + 1))
            show_image(reconstruct_from_PCA(Z, Ul, mu)[i])
            print(im_name)
            plt.savefig("../../images/{}".format(im_name))'''
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part 2d-2f: cluster toy dataset
    np.random.seed(1234)
    pts = generate_points_2d(20)
    #print("Using kmeans and random_init")
    #kMeans(pts, 3, plot=True)
    #print("Using kmedoids and random_init")
    #kMedoids(pts, 3, plot=True)
    #print("Using kmeans and cheat_init")
    #kMeans(pts, 3, init="cheat", plot=True)
    #print("Using kmedoids and cheat_init")
    #kMedoids(pts, 3, init="cheat", plot=True)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###    
    # part 3a: cluster faces
    '''np.random.seed(1234)
    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)

    kmeans_scores = []
    kmeans_runtime = []
    kmeds_scores = []
    kmeds_runtime = []
    for i in range(0, 10):
        start = time.time()
        kmeans_clusterset = kMeans(points, 4, plot=False)
        end = time.time()
        kmeans_runtime.append(end - start)
        kmeans_score = kmeans_clusterset.score()
        kmeans_scores.append(kmeans_score)

        start = time.time()
        kmeds_clusterset = kMedoids(points, 4, plot=False)
        end = time.time()
        kmeds_runtime.append(end - start)
        kmeds_score = kmeds_clusterset.score()
        kmeds_scores.append(kmeds_score)
    
    print("kmeans average score: {}".format(sum(kmeans_scores) / float(len(kmeans_scores))))
    print("kmeans max score: {}".format(max(kmeans_scores)))
    print("kmeans min score: {}".format(min(kmeans_scores)))
    print("kmeans average runtime: {}s".format((sum(kmeans_runtime) / float(len(kmeans_runtime)))))
    print("kmeans max runtime: {}".format(max(kmeans_runtime)))
    print("kmeans min runtime: {}".format(min(kmeans_runtime)))
    print("kmeds average score: {}".format(sum(kmeds_scores) / float(len(kmeds_scores))))
    print("kmeds max score: {}".format(max(kmeds_scores)))
    print("kmeds min score: {}".format(min(kmeds_scores)))
    print("kmeds average runtime: {}s".format((sum(kmeds_runtime) / float(len(kmeds_runtime)))))
    print("kmeds max runtime: {}".format(max(kmeds_runtime)))
    print("kmeds min runtime: {}".format(min(kmeds_runtime)))'''

    # part 3b: explore effect of lower-dimensional representations on clustering performance
    '''np.random.seed(1234)
    X2, y2 = util.limit_pics(X, y, [4, 13], 40)

    li = []
    for i in range(1, 43, 2):
        li.append(i)
    
    kmeans_face_scores = []
    kmeds_face_scores = []
    for l in li:
        print(l)
        Z, Ul = apply_PCA_from_Eig(X2, U[0], l, mu)
        points2 = build_face_image_points(Z, y2)
        kmeans_face_cset = kMeans(points2, 2, init="cheat", plot=False)
        kmeans_face_scores.append(kmeans_face_cset.score())
        kmeds_face_cset = kMedoids(points2, 2, init="cheat", plot=False)
        kmeds_face_scores.append(kmeds_face_cset.score())
    
    plt.plot(li, kmeans_face_scores, label="K-Means")
    plt.plot(li, kmeds_face_scores, label="K-Medoids")
    plt.title("Clustering Score Vs. Number of Principal Components")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Score")
    plt.legend()
    plt.show()'''

    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
    np.random.seed(1234)
    
    best_score = float("-inf")
    best_im = [0, 0]
    worst_score = float("inf")
    worst_im = [0 ,0]
    for i in range(0, 19):
        for j in range(0, 19):
            if i != j:
                X3, y3 = util.limit_pics(X, y, [i, j], 40)
                curr_points = build_face_image_points(X3, y3)
                c_set = kMedoids(curr_points, 2, init="cheat", plot=False)
                curr_score = c_set.score()
                if curr_score > best_score:
                    best_score = curr_score
                    best_im[0] = i
                    best_im[1] = j
                
                if curr_score < worst_score:
                    worst_score = curr_score
                    worst_im[0] = i
                    worst_im[1] = j
    
    print("The Most Discriminative Images were {}, with a score of {}".format(best_im, best_score))
    plot_representative_images(X, y, best_im, title="Most Discriminative")
    print("The Least Discriminative Images {}, with a score of {}".format(worst_im, worst_score))
    plot_representative_images(X, y, worst_im, title="Least Discriminative")
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
