'''Gaussian mixture model
train GMM for 14 samples of every number
calculate the posterior of last 2 samples for every number
and choose the greatest one as result.
'''
# Author: Chen Xi <2017202068@ruc.edu.cn>


# coding = UTF-8

import numpy as np 
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import pickle
from numpy.random import RandomState
from utils import get_mfccs, load_data_and_normalize

def init(X, n_components = 1, init_params = 'kmeans', random_state = 0):
    '''Initiate the posterior probability matrix P.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data array.

    n_components : int, defaults to 1

    init_params : string
        The method of initiation.

    random_state : int， defaults to 0
        The random_state.

    Returns
    -------
    P : array-like, shape ( n_samples, n_components)
        The posterior probability matrix.
    '''

    n_samples, _ = X.shape 
    random_state = RandomState(random_state)

    # Initiate the posterior probability matrix P.
    if init_params == 'kmeans':
        P = np.zeros((n_samples, n_components))
        label = KMeans(n_clusters=n_components, n_init=2).fit(X).labels_
        P[np.arange(n_samples), label] = 1
    elif init_params == 'random':
        P = random_state.rand(n_samples, n_components)
        P /= P.sum(axis=1)[:, np.newaxis]
    else:
        raise ValueError("Unimplemented initialization method '%s'"
                            % init_params)
    
    return P

def estimate_gaussian_parameters(X, P):
    '''Use posterioir matrix P to estimate gaussian parameters: weights, means, covariances.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    P : array-like, shape (n_samples, n_components)

    Returns
    -------
    weights : array, shape (n_components)

    means : array-like, shape (n_components, n_features)

    covariances : array-like, shape (n_components, n_features)

    '''
    n_samples, n_components = P.shape
    _, n_features = X.shape
    weights = P.sum(axis=0)/n_samples

    # means = np.dot(P.T, X)/np.array([P.sum(axis=0)]).T
    # covariances = np.dot()

    means = np.zeros((n_components, n_features), dtype=float)
    for i in range(n_components):
        for j in range(n_samples):
            means[i] += X[j]*P[j][i]
        means[i] /= np.sum(P[:, i])

    covariances = np.zeros((n_components, n_features), dtype=float)
    for i in range(n_components):
        for j in range(n_samples):
            covariances[i] +=  P[j][i]*(X[j]-means[i])*(X[j]-means[i])
        covariances[i] /= np.sum(P[:, i])

    return weights, means, covariances

def expectation(weights, means, covariances, X):
    '''Use model parameters to calculate the likelihood of every points 
    and return the matrix of posterior probability.

    Parameters
    ----------
    weights : array, shape (n_components, )

    means : array-like, shape (n_components, n_features)

    covariances : array-like, shape (n_components, n_features)

    X : array-like, shape (n_samples, n_features)

    Returns
    -------
    P : array-like, shape (n_samples, n_components)
        The posterior probability matrix.
    '''
    n_components, _ = means.shape
    n_samples, _ = X.shape
    P = np.zeros((n_samples, n_components))
    for sample in range(n_samples):
        for component in range(n_components):
            var = multivariate_normal(mean = means[component], cov = covariances[component])
            P[sample][component] = var.pdf(X[sample])*weights[component]

    for i in range(n_samples):
        P[i] = P[i]/np.sum(P[i])

    return P

def maximization(X, P):
    '''Use posterior probability to renew model parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    P : array-like, shape (n_samples, n_components)

    Returns
    -------
    weights : array, shape (n_components, )

    means : array-like, shape (n_components, n_features)

    covariances : array-like, shape (n_components, n_features)

    '''
    weights, means, covariances = estimate_gaussian_parameters(X, P)

    return weights, means, covariances


def score(weights, means, covariances, X):
    '''Calculate the log-likelihood of GMM.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)

    covariances : array-like, shape (n_components, n_features)

    X :

    Returns
    -------
    log_likelihood : float
    '''  
    n_components = means.shape[0]
    log_likelihood = 0.0

    for x in X:
        p = 0.0
        for i in range(n_components):
            var = multivariate_normal(mean = means[i], cov = covariances[i])
            p = var.pdf(x)
        log_likelihood += np.log(p)

    return log_likelihood

def GMM(weights, means, covariances, X, max_iter = 20, threshold = 1e-3):
    '''An implementation of Gaussian mixture model using EM algorithm.
    
    Parameters
    ----------
    weights :

    means :

    covariances :
    
    X :

    max_iter :

    threshold : float , defaults to 1e-3

    Returns
    -------
    mean: array, shape (n_features, )
        The trained mean vector.
    covariance: diag matrix, shape (n_features, n_features)
        The trained diagonal covariance array.
    '''

    # train parameters by E-step and M-step
    iter = 0
    perf = 1.0

    while iter < max_iter and perf > threshold :
        P = expectation(weights, means, covariances, X)
        n_weights, means, covariances = maximization(X, P)
        
        perf = np.linalg.norm(n_weights-weights)

        weights = n_weights
        iter += 1
    
    return n_weights, means, covariances

def test(models):
    '''Use trained GMMs to calculate accuracy
    
    Parameters
    ----------
    models : list, shape (n_model, 2)
        n_models == 10.
        means, covariances.

    Returns
    -------
    accuracy : float
        The number of
    '''
    n_model = len(models)
    for i in range(n_model):   
        posterior = []
        accuracy = 0

        for num in range(15, 17):
            root = "records/digit_" + str(i)
            url = root + "/" + str(num) + "_" + str(i) + ".wav"
            mfccs = get_mfccs(url)
            X = mfccs.T

            for m in range(n_model):
                p = score(models[m][0], models[m][1], models[m][2], X)
                posterior.append(p)
            
            if posterior.index(max(posterior)) == i:
                accuracy += 1

        accuracy /= 2
        print("the accuracy of '%d' is '%.2f'%s." %(i,accuracy*100, "%"))

def bic(X, weights, means, covariances):
    '''Bayesian information criterion for the current model on the input X.
    
    Parameters
    ----------
    X : array of shape (n_samples, n_dimensions)

    n_components : int

    Returns
    -------
    bic : float
        The lower the better.
    '''
    n_components, n_features = means.shape
    log_likelihood = score(weights, means, covariances, X)

    cov_params = n_components * n_features
    mean_params = n_components * n_features
    n_parameters = cov_params + mean_params

    return (-2 * log_likelihood + n_parameters*np.log(X.shape[0]))

def choose_model(X, max_components=10):
    '''Use BIC to choose appropriate n_components
    
    Parameters
    ----------
    X :

    max_components : int, defaults to 10

    Returns
    -------
    i : int
    '''
    min_bic = float('inf') # float最大值
    best_components = -1
    for i in range(2, max_components+1):
        # train using i components
        P = init(X, i)
        weights, means, covariances = estimate_gaussian_parameters(X, P)
        weights, means, covariances = GMM(weights, means, covariances, X)

        bic_ = bic(X, weights, means, covariances)
        if bic_ < min_bic:
            min_bic = bic_
            best_components = i

    print("the best component is %d" %(best_components))

    P = init(X, best_components)
    weights, means, covariances = estimate_gaussian_parameters(X, P)
    weights, means, covariances = GMM(weights, means, covariances, X)
    
    return weights, means, covariances

if __name__ == "__main__":

    try:
        f = open('models.txt','rb') 
        models = pickle.load(f)
        f.close()
        test(models)

    except FileNotFoundError:
        print("File models.txt doesn't exist.\nTraining is going on...")
        models = []

        for i in range(10): 
            X = load_data_and_normalize(i)

            '''n_components = 2
            P = init(X, n_components)
            weights, means, covariances = estimate_gaussian_parameters(X, P)
            mean, covariance = GMM(weights, means, covariances, X)
            '''

            weights, means, covariances = choose_model(X, 6) 
            print("The model of num %i has been trained." %i)
            models.append([])
            models[i].append(weights)
            models[i].append(means)
            models[i].append(covariances)
        
        f = open('models.txt','wb') 
        pickle.dump(models, f, -1)
        f.close()
        print("Trained models have been stored into models.txt.\nTest is going on...")

        test(models)

        

        

