## baseline method: Adversarial Sphere (linear hyperplane)

import load_data as ld
import numpy as np 
from sklearn import decomposition

import setproctitle
import math
import os

if __name__ == "__main__":
    args = ld.argparser(dataset='mnist', metric='euclidean', epsilon=1.58, alpha=0.01) 
    # args = ld.argparser(dataset='cifar', metric='euclidean', epsilon=0.2453, alpha=0.05)
    setproctitle.setproctitle('python')

    print('dataset: {dataset}\t\t' 'metric: {metric}\t\t' 
            'epsilon: {epsilon}\t\t' 'alpha: {alpha}\t\t'.format(
            dataset=args.dataset, metric=args.metric, 
            epsilon=args.epsilon, alpha=args.alpha))

    #### load the datasets
    if args.dataset == 'mnist':
        train_loader, test_loader, valid_loader = ld.mnist_loaders(path='./data/'+args.dataset, 
                                                                seed=args.seed,
                                                                ratio=args.ratio)
        for i, (X,y) in enumerate(train_loader):
            train_data = X.view(-1, 28*28).numpy()
        for i, (X,y) in enumerate(test_loader):
            test_data = X.view(-1, 28*28).numpy()

    elif args.dataset == 'cifar':
        train_loader, test_loader, valid_loader = ld.cifar_loaders(path='./data/'+args.dataset, 
                                                                seed=args.seed, 
                                                                ratio=args.ratio)
        for i, (X,y) in enumerate(train_loader):
            train_data = X.view(-1, 3*32*32).numpy()
        for i, (X,y) in enumerate(test_loader):
            test_data = X.view(-1, 3*32*32).numpy()	

    elif args.dataset== 'fmnist':
        train_loader, test_loader, valid_loader = ld.fashion_mnist_loaders(path='./data/'+args.dataset, 
                                                                            seed=args.seed, 
                                                                            ratio=args.ratio)
        for i, (X,y) in enumerate(train_loader):
            train_data = X.view(-1, 28*28).numpy()
        for i, (X,y) in enumerate(test_loader):
            test_data = X.view(-1, 28*28).numpy()	            	

    elif args.dataset== 'svhn':
        train_loader, test_loader, valid_loader = ld.svhn_loaders(path='./data/'+args.dataset, 
                                                                    seed=args.seed, 
                                                                    ratio=args.ratio)
        for i, (X,y) in enumerate(train_loader):
            train_data = X.view(-1, 3*32*32).numpy()
        for i, (X,y) in enumerate(test_loader):
            test_data = X.view(-1, 3*32*32).numpy()

    else:
        raise ValueError('specified dataset name not recognized.')

    #### compute the coefficient (top principal component) for hyperplane
    pca = decomposition.PCA(n_components=100, svd_solver='full')
    pca.fit(train_data)
    w = pca.components_[0]      # set the first principal component as w
    w_norm = np.sqrt(np.sum(w**2))

    n_train = train_data.shape[0]
    b_arr = np.sort(np.dot(train_data, w))[::-1]
    print('sorted projected value:', b_arr)
    b = b_arr[math.ceil(n_train*args.alpha)]
    print('threshold for error region:', b)

    #### compute the measure and expansion of error region
    risk_train = np.sum(b_arr>b)/n_train
    advRisk_train = np.sum(b_arr>b-args.epsilon*w_norm)/n_train
    print('')
    print('Risk for train data: ', '{:.2%}'.format(risk_train))
    print('Adversarial Risk for train data: ', '{:.2%}'.format(advRisk_train))

    #### test the generalization on testing data
    n_test = len(test_data)
    b_arr_test = np.sort(np.dot(test_data, w))[::-1]

    risk_test = np.sum(b_arr_test>b)/float(n_test)
    advRisk_test = np.sum(b_arr_test>b-args.epsilon*w_norm)/float(n_test)
    print('Risk for test data: ', '{:.2%}'.format(risk_test))
    print('Adversarial Risk for test data: ', '{:.2%}'.format(advRisk_test))
