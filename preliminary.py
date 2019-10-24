#### exploratory data analysis
#### obtain the k-nearest-neighbour distance estimates on the training dataset
#### (adapt the implementation of constructing knn graph function in DeBaCl)

import load_data as ld
import numpy as np
import setproctitle
import math
import os

## soft dependencies for knn
try:
    import scipy.spatial.distance as _spd
    import scipy.special as _spspec
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False

try:
    import sklearn.neighbors as _sknbr
    _HAS_SKLEARN = True
except:
    _HAS_SKLEARN = False
    
    
#### knn graph construction (adapted from DeBaCl library)
def knn_graph(X, k, method='brute_force', leaf_size=30, metric='euclidean'):
    n, p = X.shape
    if method == 'kd_tree':
        if _HAS_SKLEARN:
            kdtree = _sknbr.KDTree(X, leaf_size=leaf_size, metric=metric)
            distances, neighbors = kdtree.query(X, k=k, return_distance=True,
                                                sort_results=True)
            radii = distances[:, -1]
        else:
            raise ImportError("The scikit-learn library could not be loaded." +
                              " It is required for the 'kd-tree' method.")

    if method == 'ball_tree':
        if _HAS_SKLEARN:
            btree = _sknbr.BallTree(X, leaf_size=leaf_size, metric=metric)
            distances, neighbors = btree.query(X, k=k, return_distance=True,
                                               sort_results=True)
            radii = distances[:, -1]
        else:
            raise ImportError("The scikit-learn library could not be loaded." +
                              " It is required for the 'ball-tree' method.")

    else:  # assume brute-force
        if not _HAS_SCIPY:
            raise ImportError("The 'scipy' module could not be loaded. " +
                              "It is required for the 'brute_force' method " +
                              "for building a knn similarity graph.")

        d = _spd.pdist(X, metric=metric)
        D = _spd.squareform(d)
        rank = _np.argsort(D, axis=1)
        neighbors = rank[:, 0:k]
        k_nbr = neighbors[:, -1]
        radii = D[_np.arange(n), k_nbr]
        
    return neighbors, radii
    
if __name__ == "__main__":
    args = ld.argparser(dataset='mnist', metric='infinity', k=50)
    # args = ld.argparser(dataset='cifar', metric='euclidean', alpha=0.05)
    setproctitle.setproctitle('preliminary')

    #### load the datasets
    if args.dataset== 'mnist':
        train_loader, test_loader, valid_loader = ld.mnist_loaders(path='./data/'+args.dataset, 
                                                                    seed=args.seed, 																	    
                                                                    ratio=args.ratio)
        for i, (X,y) in enumerate(train_loader):
            train_data = X.view(-1, 28*28).numpy()
        for i, (X,y) in enumerate(test_loader):
            test_data = X.view(-1, 28*28).numpy()

    elif args.dataset== 'fmnist':
        train_loader, test_loader, valid_loader = ld.fashion_mnist_loaders(path='./data/'+args.dataset, 
                                                                            seed=args.seed, 
                                                                            ratio=args.ratio)
        for i, (X,y) in enumerate(train_loader):
            train_data = X.view(-1, 28*28).numpy()
        for i, (X,y) in enumerate(test_loader):
            test_data = X.view(-1, 28*28).numpy()

    elif args.dataset== 'cifar':
        train_loader, test_loader, valid_loader = ld.cifar_loaders(path='./data/'+args.dataset, 
                                                                    seed=args.seed, 
                                                                    ratio=args.ratio)
        for i, (X,y) in enumerate(train_loader):
            train_data = X.view(-1, 3*32*32).numpy()
        for i, (X,y) in enumerate(test_loader):
            test_data = X.view(-1, 3*32*32).numpy()

    elif args.dataset== 'svhn':
        train_loader, test_loader, valid_loader = ld.svhn_loaders(path='./data/'+args.dataset, 
                                                                    seed=args.seed, 
                                                                    ratio=args.ratio)
        for i, (X,y) in enumerate(train_loader):
            train_data = X.view(-1, 3*32*32).numpy()
        for i, (X,y) in enumerate(test_loader):
            test_data = X.view(-1, 3*32*32).numpy()
            
    else:
        raise ValueError('Specified dataset name not recognized')

    if not os.path.exists('./eda'):
        os.makedirs('./eda')

    #### get the k nearest neighbour distance estimates on training dataset    
    if args.metric == 'infinity':
        print('=============== get knn graph (k = ' + str(args.k) + ') ===============')
        savepath = './eda/knn_'+args.dataset+'_'+args.metric+'_'+str(args.k)+'.txt'
        ## save the radius (k-th nearest neighbor) for each example
        if not os.path.exists(savepath):
            _, radii = knn_graph(train_data, k=args.k, method='ball-tree', metric='cityblock')
        else:
            print("radii file already saved for L-infinity case.")
        
    elif args.metric == 'euclidean':
        args.k = math.ceil(train_data.shape[0]*args.alpha)*2        # tolerance for more neighbors   
        print('=============== get knn graph (k = ' + str(args.k) + ') ===============')
        savepath = './eda/knn_'+args.dataset+'_'+args.metric+'_'+str(args.k)+'_neighbors.npy'
        ## save the nearest neighbors for each example
        if not os.path.exists(savepath):
            neighbors, _ = knn_graph(train_data, k=args.k, method='ball-tree')
        else:
            print("neighbors file already saved for L-2 case.")

    else:
        raise ValueError('Unknown type of metric')

