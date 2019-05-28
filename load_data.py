#### load image dataset and define input arguments

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td

import argparse
import parser

#### MNIST
def mnist_loaders(path, ratio, seed=None): 
    mnist_train = datasets.MNIST(path, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(path, train=False, download=True, transform=transforms.ToTensor())

    if ratio == 0:
        train_loader = td.DataLoader(mnist_train_total, batch_size=len(mnist_train))
        test_loader = td.DataLoader(mnist_test, batch_size=len(mnist_test))
        return train_loader, test_loaders

    elif ratio>0 and ratio<1:   
        torch.manual_seed(seed)
        num_valid = int(ratio*len(mnist_train))    
        _, mnist_valid = td.random_split(mnist_train, [len(mnist_train)-num_valid, num_valid])    

        train_loader = td.DataLoader(mnist_train, batch_size=len(mnist_train))
        test_loader = td.DataLoader(mnist_test, batch_size=len(mnist_test))
        valid_loader = td.DataLoader(mnist_valid, batch_size=num_valid)
        return train_loader, test_loader, valid_loader

    else: 
        raise ValueError('ratio should be in range [0,1).')


#### Fashion-MNIST
def fashion_mnist_loaders(path, ratio, seed=None): 
    fmnist_train = datasets.FashionMNIST(path, train=True,
                        download=True, transform=transforms.ToTensor())
    fmnist_test = datasets.FashionMNIST(path, train=False,
                        download=True, transform=transforms.ToTensor())
    if ratio == 0:
        train_loader = td.DataLoader(fmnist_train, batch_size=batch_size)
        test_loader = td.DataLoader(fmnist_test, batch_size=batch_size)
        return train_loader, test_loader

    elif ratio>0 and ratio<1:
        torch.manual_seed(seed)
        num_valid = int(ratio*len(fmnist_train))
        _, fmnist_valid = td.random_split(fmnist_train, [len(fmnist_train)-num_valid, num_valid])    

        train_loader = td.DataLoader(fmnist_train, batch_size=len(fmnist_train))
        test_loader = td.DataLoader(fmnist_test, batch_size=len(fmnist_test))
        valid_loader = td.DataLoader(fmnist_valid, batch_size=num_valid)
        return train_loader, test_loader, valid_loader  

    else: 
        raise ValueError('ratio should be in range [0,1).')


#### CIFAR-10
def cifar_loaders(path, ratio, seed=None): 

    cifar_train = datasets.CIFAR10(path, train=True, download=True, 
                    transform=transforms.Compose([transforms.ToTensor(),]))                                  
    cifar_test = datasets.CIFAR10(path, train=False, 
                    transform=transforms.Compose([transforms.ToTensor(),]))      

    if ratio == 0:
        train_loader = td.DataLoader(cifar_train, batch_size=len(cifar_train))
        test_loader = td.DataLoader(cifar_test, batch_size=len(cifar_test))
        return train_loader, test_loader
    
    elif ratio>0 and ratio<1:     
        # train-validation split based on the value of ratio
        torch.manual_seed(seed)
        num_valid = int(ratio*len(cifar_train))
        _, cifar_valid = td.random_split(cifar_train, [len(cifar_train)-num_valid, num_valid])    

        train_loader = td.DataLoader(cifar_train, batch_size=len(cifar_train))
        test_loader = td.DataLoader(cifar_test, batch_size=len(cifar_test))
        valid_loader = td.DataLoader(cifar_valid, batch_size=num_valid)
        return train_loader, test_loader, valid_loader

    else: 
        raise ValueError('ratio should be in range [0,1).')


#### SVHN
def svhn_loaders(path, ratio, seed=None): 
    svhn_train = datasets.SVHN(path, split='train', download=True, 
                    transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    svhn_test = datasets.SVHN(path, split='test', download=True, 
                    transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    if ratio == 0:
        train_loader = td.DataLoader(svhn_train, batch_size=batch_size)
        test_loader = td.DataLoader(svhn_test, batch_size=batch_size)
        return train_loader, test_loader

    elif ratio>0 and ratio<1:
        torch.manual_seed(seed)
        num_valid = int(ratio*len(svhn_train))
        _, svhn_valid = td.random_split(svhn_train, [len(svhn_train)-num_valid, num_valid])    

        train_loader = td.DataLoader(svhn_train, batch_size=len(svhn_train))
        test_loader = td.DataLoader(svhn_test, batch_size=len(svhn_test))
        valid_loader = td.DataLoader(svhn_valid, batch_size=num_valid)
        return train_loader, test_loader, valid_loader

    else: 
        raise ValueError('ratio should be in range [0,1).')


#### define the argparser for simplicity
def argparser(dataset=None, metric=None, epsilon=0.3, k=50, q=0.8, clusters=20,
              iter=30, repeat=10, alpha=0.01, seed=0, ratio=0.2, verbose=2000, delta=0.01):
    parser = argparse.ArgumentParser()

    # image dataaset
    parser.add_argument('--dataset', type=str, default=dataset, help='dataset name')
    parser.add_argument('--seed', type=int, default=seed, help='random seed number')
    parser.add_argument('--ratio', type=float, default=ratio, help='ratio of validation dataset')

    # perturbation family
    parser.add_argument('--metric', type=str, default=metric, help='type of perturbations')
    parser.add_argument('--epsilon', type=float, default=epsilon, help='perturbation strength')	

    # k nearest neighbour
    parser.add_argument('--k', type=int, default=k, help='number of nearest neighbors for knn')
    parser.add_argument('--q', type=float, default=q, help='initial covered density quantile')

    # kmeans clustering
    parser.add_argument('--clusters', type=int, default=clusters, help='number of clusters for kmeans')
    parser.add_argument('--iter', type=int, default=iter, help='number of iterations for kmeans')
    parser.add_argument('--repeat', type=int, default=repeat, help='number of repeated experiments')
    
    # other arguments
    parser.add_argument('--alpha', type=float, default=alpha, help='risk threshold')
    parser.add_argument('--delta', type=float, default=delta)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--proctitle', type=str, default="")
   
    args = parser.parse_args()

    if args.dataset is not None:
        args.proctitle += args.dataset+'/'

        if args.metric is not None:
            args.proctitle += args.metric+'/'+'epsilon_'+str(args.epsilon)

            banned = ['proctitle', 'dataset', 'metric', 'epsilon', 
                    'alpha', 'seed', 'ratio', 'k', 'iter', 'repeat', 
                    'verbose', 'delta']    
            if metric == 'euclidean':
                banned += 'q'

            for arg in sorted(vars(args)): 
                if arg not in banned and getattr(args,arg) is not None: 
                        args.proctitle += '_' + arg + '_' + str(getattr(args, arg))
        else:
            raise ValueError('Need to specify the family of perturbations.')

    else:
        raise ValueError('Need to specify the image dataset.')

    return args


def replace_10_with_0(y): 
    return y % 10   