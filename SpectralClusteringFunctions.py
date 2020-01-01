import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


def compute_similarity(X: list, sigma: float, epsilon: float):
    n = np.shape(X)[0]
    dist = []
    for i in range(n):
        for j in range(n):
            dist.append(-np.abs(X[i]-X[j]))
    aux2 = [i/(2*sigma**2) for i in dist]        
    aux1 = np.exp(aux2)
    distances = np.array(aux1).reshape(n,n)
    distances[distances < epsilon] = 0
    return distances


def compute_laplacian(adjacency: np.ndarray, normalize: bool):
    """ Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diagonal = np.sum(adjacency,axis=1)
    D = np.diag(diagonal)
    laplacian = D - adjacency
    if normalize == False:
        return laplacian
    else:
        diagonal[diagonal==0]=1e-12
        D_ = np.diag(np.sqrt(1/np.array(diagonal)))
        return (D_.dot(laplacian)).dot(D_)   
    
def spectral_decomposition(laplacian: np.ndarray):
    """ Return:
        lamb (np.array): eigenvalues of the Laplacian
        U (np.ndarray): corresponding eigenvectors.
    """
    ### Eigenvalues and eigenvectors computation
    lamb, U = np.linalg.eig(laplacian)
    ### Sorting of the eignvectors 
    idx = np.argsort(lamb)  
    lamb = lamb[idx]
    U = U[:,idx]
    
    return lamb, U
    # Your code here
    
def compute_number_connected_components(lamb: np.array, threshold: float):
    """ lamb: array of eigenvalues of a Laplacian
        Return:
        n_components (int): number of connected components.
    """
    a = np.sum(lamb<threshold)
    return a

class SpectralClustering():
    def __init__(self, n_classes: int, normalize: bool):
        self.n_classes = n_classes
        self.normalize = normalize
        self.laplacian = None
        self.e = None
        self.U = None
        self.clustering_method =  KMeans(n_clusters=self.n_classes) # Your code here
        
    def fit_predict(self, adjacency):
        """ Your code should be correct both for the combinatorial
            and the symmetric normalized spectral clustering.
            Return:
            y_pred (np.ndarray): cluster assignments.
        """
        ### Compute Laplacian Matrix
        self.laplacian = compute_laplacian(adjacency, self.normalize)
        ### Compute Eigenvalues and Eigenvectors
        self.e , self.U = spectral_decomposition(self.laplacian)
        ### Select the number of eigenvectors required for the n_classes
        aux = self.U[:,:self.n_classes]
        
        Y = np.zeros([len(adjacency),self.n_classes])
        
        if self.normalize == True:
            for i in range(len(adjacency)):
                norm_= np.linalg.norm(aux[i])
                if norm_!=0 :
                    for j in range(self.n_classes):
                        Y[i,j] = aux[i,j]/norm_
        else:
            Y = aux
            
        kmeans = self.clustering_method.fit(Y)
        y_pred = kmeans.labels_  
        # Your code here
        return y_pred
    
'''
   
def epsilon_similarity_graph(X: np.ndarray, sigma=None, epsilon=None):
    """ X (n x d): coordinates of the n data points in R^d.
        sigma (float): width of the kernel
        epsilon (float): threshold
        Return:
        adjacency (n x n ndarray): adjacency matrix of the graph.
    """
    # Your code here
    distances = squareform(np.exp(-pdist(X,'sqeuclidean')/2*sigma**2))
    distances[distances < epsilon] = 0

    return distances    
'''     