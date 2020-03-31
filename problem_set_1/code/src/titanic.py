"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        temp = Counter(y).most_common(2)
        value_1 = temp[0][0]
        count_1 = float(temp[0][1])
        value_2 = temp[1][0]
        count_2 = float(temp[1][1])
        p_1 = count_1 / y.size
        p_2 = count_2 / y.size
        self.probabilities_ = [ p_1, p_2 ]
        ### ========== TODO : END ========== ###
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        
        y = np.random.choice(2, len(X), p=self.probabilities_)
        
        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in xrange(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    test_tot = 0.
    train_tot = 0.

    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        clf.fit(X_train, y_train)
        test_pred = clf.predict(X_test)
        train_pred = clf.predict(X_train)
        train_err = 1 - metrics.accuracy_score(y_train, train_pred, normalize=True)
        test_err = 1 - metrics.accuracy_score(y_test, test_pred, normalize=True)
        test_tot += test_err
        train_tot += train_err

    train_error = train_tot / ntrials
    test_error = test_tot / ntrials
        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    #print 'Plotting...'
    #for i in xrange(d) :
    #    plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print 'Classifying using Majority Vote...'
    mclf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    mclf.fit(X, y)                  # fit training data using the classifier
    y_pred = mclf.predict(X)        # take the classifier and run it on the training data
    mv_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % mv_error
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print 'Classifying using Random...'
    rclf = RandomClassifier()
    rclf.fit(X, y)
    y_pred = rclf.predict(X)
    rand_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % rand_error
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print 'Classifying using Decision Tree...'
    dclf = DecisionTreeClassifier(criterion="entropy")
    dclf.fit(X, y)
    y_pred = dclf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    ### ========== TODO : END ========== ###
    
    
    
    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """
    
    
    
    ### ========== TODO : START ========== ###
    # part d: use cross-validation to compute average training and test error of classifiers
    print 'Investigating various classifiers...'
    majority_vote_err = error(mclf, X, y)
    print('\t-- Majority Vote Classifier training error:{}, test error: {}'.format(majority_vote_err[0], majority_vote_err[1]))
    random_err = error(rclf, X, y)
    print('\t-- Random Classifier training error:{}, test error: {}'.format(random_err[0], random_err[1]))
    decision_tree_err = error(dclf, X, y)
    print('\t-- Decision Tree Classifier training error:{}, test error: {}'.format(decision_tree_err[0], decision_tree_err[1]))
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: investigate decision tree classifier with various depths
    print 'Investigating depths...'
    max_depths = []
    train_errors = []
    test_errors = []
    mv_errors = np.array([mv_error] * 20)
    rand_errors = np.array([rand_error] * 20)
    for i in range(1, 21):
        dtclf = DecisionTreeClassifier(criterion="entropy", max_depth=i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        dtclf.fit(X_train, y_train)
        test_pred = dtclf.predict(X_test)
        train_pred = dtclf.predict(X_train)
        test_error = 1 - metrics.accuracy_score(y_test, test_pred, normalize=True)
        train_error = 1 - metrics.accuracy_score(y_train, train_pred, normalize=True)
        max_depths.append(i)
        train_errors.append(train_error)
        test_errors.append(test_error)

    max_depths = np.array(max_depths)
    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)

    plt.scatter(max_depths, train_errors, label="Decision Tree Classifier Train Set")
    dt_train_results = np.polyfit(max_depths, train_errors, 2)
    plt.plot(max_depths, dt_train_results[0] * (max_depths ** 2) + dt_train_results[1] * max_depths + dt_train_results[2])

    plt.scatter(max_depths, test_errors, label="Decision Tree Classifier Test Set")
    dt_test_results = np.polyfit(max_depths, test_errors, 2)
    plt.plot(max_depths, dt_test_results[0] * (max_depths ** 2) + dt_test_results[1] * max_depths + dt_test_results[2]) 

    plt.scatter(max_depths, mv_errors, label="Majority Vote Classifier")
    mv_results = np.polyfit(max_depths, mv_errors, 1)
    plt.plot(max_depths, mv_results[0] * max_depths + mv_results[1])

    plt.scatter(max_depths, rand_errors, label="Random Classifier")
    r_results = np.polyfit(max_depths, rand_errors, 1)
    plt.plot(max_depths, r_results[0] * max_depths + r_results[1])

    plt.title("Error Rates with Different Classifiers and their Hyperparameters")
    plt.xlabel("Max Depth (for Decision Tree Classifier only)")
    plt.ylabel("Error Rates")
    plt.legend()
    ax = plt.subplot(111)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8))
    plt.show()
    #plt.savefig("classifier_error_rates.png")
    plt.clf()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part f: investigate decision tree classifier with various training set sizes
    print 'Investigating training set sizes...'
    dclf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    train_sizes = []
    test_errors = []
    train_errors = []
    m_errors = []
    r_errors = []
    for i in range(1, 20):
        train_size = float(i) * 0.05
        train_sizes.append(train_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1. - train_size))
        dclf.fit(X_train, y_train)
        
        test_pred = dclf.predict(X_test)
        train_pred = dclf.predict(X_train)

        m_pred = mclf.predict(X_test)
        r_pred = rclf.predict(X_test)

        m_error = 1 - metrics.accuracy_score(y_test, m_pred, normalize=True)
        r_error = 1 - metrics.accuracy_score(y_test, r_pred, normalize=True)
        train_err = 1 - metrics.accuracy_score(y_train, train_pred, normalize=True)
        test_err = 1 - metrics.accuracy_score(y_test, test_pred, normalize=True)

        test_errors.append(test_err)
        train_errors.append(train_err)
        m_errors.append(m_error)
        r_errors.append(r_error)
    
    train_sizes = np.array(train_sizes)
    test_errors = np.array(test_errors)
    train_errors = np.array(train_errors)
    m_errors = np.array(m_errors)
    r_errors = np.array(r_errors)

    plt.scatter(train_sizes, train_errors, label="Decision Tree Train Set Errors")
    dt_train_results = np.polyfit(train_sizes, train_errors, 2)
    plt.plot(train_sizes, dt_train_results[0] * (train_sizes) ** 2 + dt_train_results[1] * train_sizes + dt_train_results[2])

    plt.scatter(train_sizes, test_errors, label="Decision Tree Test Set Errors")
    dt_test_results = np.polyfit(train_sizes, test_errors, 2)
    plt.plot(train_sizes, dt_test_results[0] * (train_sizes) ** 2 + dt_test_results[1] * train_sizes + dt_test_results[2])

    plt.scatter(train_sizes, m_errors, label="Majority Vote Classifier Test Set Errors")
    mv_results = np.polyfit(train_sizes, m_errors, 1)
    plt.plot(train_sizes, mv_results[0] * train_sizes + mv_results[1])

    plt.scatter(train_sizes, r_errors, label="Random Classifier Test Set Errors")
    r_results = np.polyfit(train_sizes, r_errors, 1)
    plt.plot(train_sizes, r_results[0] * train_sizes + r_results[1])

    plt.title("Error Rates with Different Training Set Sizes")
    plt.xlabel("Training Set Sizes (Proportion of the Dataset)")
    plt.ylabel("Error Rates")
    ax = plt.subplot(111)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8))
    plt.show()
    ### ========== TODO : END ========== ###
    
       
    print 'Done'


if __name__ == "__main__":
    main()
