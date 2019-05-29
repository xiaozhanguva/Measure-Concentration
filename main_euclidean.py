#### problem: empirically estimate the intrinsic robustness on real dataset 
####          (l2-norm bounded perturbations on MNIST and CIFAR-10 datasets)
####
#### strategy: cover alpha-fraction of images with T union of balls as the error region,
####           place the ball one-by-one such that each placement has minium expansion
####			(use gpus for acceleration)

import load_data as ld
import numpy as np 
import random
from pathlib import Path
import torch

import setproctitle
import time
import math
import os

if __name__ == "__main__":
	args = ld.argparser(dataset='mnist', epsilon=1.58, alpha=0.01, clusters=10, delta=0.25)
	# args = ld.argparser(dataset='cifar', epsilon=0.2453, alpha=0.05, clusters=5, delta=0)
	setproctitle.setproctitle('python')

	print('dataset: {dataset}\t\t' 'metric: {metric}\t\t'
		  'epsilon: {epsilon}\t\t' '#balls: {clusters}\t\t'
		  'alpha: {alpha}\t\t' 'delta:{delta}'.format(
			dataset=args.dataset, metric=args.metric, epsilon=args.epsilon, 
			clusters=args.clusters, alpha=args.alpha, delta=args.delta))

	#### choose to use gpu if available
	device = None
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	#### define the result file path
	res_filepath = './results/fine_tune/'+args.proctitle+'.txt'
	if not os.path.exists(os.path.dirname(res_filepath)):
		os.makedirs(os.path.dirname(res_filepath))
	print("saving file to {}".format(res_filepath))

	#### load the datasets
	if args.dataset == 'mnist':
		train_loader, test_loader, valid_loader = ld.mnist_loaders(path='./data/'+args.dataset, 
																   seed=args.seed,
																   ratio=args.ratio)
		for i, (X,y) in enumerate(train_loader):
			train_data = X.view(-1, 28*28).to(device)
		for i, (X,y) in enumerate(test_loader):
			test_data = X.view(-1, 28*28).to(device)

	elif args.dataset == 'cifar':
		train_loader, test_loader, valid_loader = ld.cifar_loaders(path='./data/'+args.dataset, 
																   seed=args.seed, 
																   ratio=args.ratio)
		for i, (X,y) in enumerate(train_loader):
			train_data = X.view(-1, 3*32*32).to(device)
		for i, (X,y) in enumerate(test_loader):
			test_data = X.view(-1, 3*32*32).to(device)	

	elif args.dataset== 'fmnist':
		train_loader, test_loader, valid_loader = ld.fashion_mnist_loaders(path='./data/'+args.dataset, 
																			seed=args.seed, 
																			ratio=args.ratio)
		for i, (X,y) in enumerate(train_loader):
			train_data = X.view(-1, 28*28).to(device)
		for i, (X,y) in enumerate(test_loader):
			test_data = X.view(-1, 28*28).to(device)

	elif args.dataset== 'svhn':
		train_loader, test_loader, valid_loader = ld.svhn_loaders(path='./data/'+args.dataset, 
																	seed=args.seed, 
																	ratio=args.ratio)
		for i, (X,y) in enumerate(train_loader):
			train_data = X.view(-1, 3*32*32).to(device)
		for i, (X,y) in enumerate(test_loader):
			test_data = X.view(-1, 3*32*32).to(device)

	else:
		raise ValueError('specified dataset name not recognized.')


	#### obatin the k-nearest neighbours for each points in the training set
	n_train = train_data.shape[0]
	args.k = math.ceil(n_train*args.alpha)*2
	savepath = './eda/knn_'+args.dataset+'_'+args.metric+'_'+str(args.k)+'_neighbors.npy'
	neighbor_ind_arr = np.load(savepath)

	res_file = open(res_filepath, "w")
	## number of points to cover (add additional points to avoid overfitting)
	n_point_tot = math.ceil(n_train*args.alpha*(1+args.delta))
	print('Total number of data points to cover:', n_point_tot)

	#### place the balls incrementally for the smallest expansion
	centroids = []
	radii = []
	index_init = []
	index_expand = []
	for t in range(args.clusters):
		print('')
		if len(index_init) >= n_point_tot:
			break
		start = time.time()

		## set the range of #covered points for the current ball
		n_lower = math.ceil((n_point_tot-len(index_init)) / (args.clusters-t))
		n_upper = n_point_tot-len(index_init)
		print('number of points to cover: ['+str(n_lower)+':'+str(n_upper)+']')

		## record the indices for remaining data points
		ind_valid = np.array([x for x in range(n_train) if x not in index_init])
		ind_valid_expand = np.array([x for x in range(n_train) if x not in index_expand])

		n_expand_opt = n_train
		iter_count = 0
		## for each training data as center, find the minimum expansion
		for i in range(n_train):
			iter_count += 1
			i_neighors = neighbor_ind_arr[i,:]
			neighbor_ind = i_neighors[~np.in1d(i_neighors, index_init)]

			## set the center at each training data point (faster, generalize better)
			center = train_data[i]
			dist = torch.sqrt(torch.sum((train_data[ind_valid]-center)**2, dim=1))
			dist_expand = torch.sqrt(torch.sum((train_data[ind_valid_expand]-center)**2, dim=1))

			## enumerate each value in [n_lower, n_upper], find the minimum expansion		
			n_point_arr = range(n_lower-1,n_upper)
			radius_arr = torch.sqrt(torch.sum((train_data[neighbor_ind[n_point_arr]]-center)**2, dim=1))

			dist_init_mat = dist.repeat(len(n_point_arr),1)-radius_arr.reshape(len(n_point_arr),1)
			count_init=torch.sum(dist_init_mat<=0, dim=1).double()

			dist_expand_mat = dist_expand.repeat(len(n_point_arr),1)-radius_arr.reshape(len(n_point_arr),1)
			count_expand = torch.sum(dist_expand_mat<=args.epsilon, dim=1).double()

			## record the minimum expansion and its index
			n_expand, index = torch.min(count_expand-count_init, 0)
			radius = radius_arr[index]

			## record the indices (initial coverage and expanded coverage)
			ind_init= ind_valid[torch.nonzero(dist <= radius).view(-1)]
			ind_expand_total = ind_valid[torch.nonzero(dist <= radius+args.epsilon).view(-1)]
			ind_expand = np.setdiff1d(ind_expand_total, index_expand)
			ratio = float(len(ind_expand)) / float(len(ind_init))

			## print the intermediate result
			if iter_count % args.verbose == 0:
				print('Iteration [{0}/{1}]\t\t' 
						'Initial: {init}\t\t' 'Expanded: {expand}\t\t'
						'Ratio: {ratio:.2f}\t\t' 'Radius: {radius:.2f}'.format(
						iter_count, n_train, 
						init=len(ind_init), expand=len(ind_expand),
						ratio=ratio, radius=radius))
		
			## record the statistics w.r.t. the smallest expansion
			if n_expand < n_expand_opt:	
				n_expand_opt = n_expand	
				init_opt = ind_init
				expand_opt = ind_expand
				center_opt = center
				radius_opt = radius
				ratio_opt = ratio	
		
		## record the optmal center and radius
		index_init.extend(init_opt)
		index_expand.extend(expand_opt)
		centroids.append(center_opt)
		radii.append(radius_opt)
		iter_time = time.time() - start

		print(' * Placed-balls [{0}/{1}]\t\t'
				'Time-elapsed: {time:.2f}\t\t'
				'Initial: {init_opt}\t\t'
				'Expanded: {expand_opt}\t\t'
				'Ratio: {ratio:.2f}\t\t'
				'Radius: {radius:.4f}'.format(
					t+1, args.clusters, time=iter_time, 
					init_opt=len(init_opt), expand_opt=len(expand_opt), 
					ratio=ratio_opt, radius=radius_opt))

	risk_train = len(index_init)/float(n_train)
	advRisk_train = len(index_expand)/float(n_train)
	print('')
	print('Risk for train data:', '{:.2%}'.format(risk_train))
	print('Adversarial risk for train data:', '{:.2%}'.format(advRisk_train))

	#### test the generalization on testing data
	count= 0
	count_expand = 0
	n_test = len(test_data)

	print('========== testing ==========')
	for idx, data in enumerate(test_data):
		if idx % args.verbose == 0:
			print('Iteration [{0}/{1}]\t\t'.format(idx, len(test_data)))

		for num, center in enumerate(centroids):
			diff = torch.sqrt(torch.sum((data-center)**2))	
			if diff <= radii[num]:
				count += 1
				break
		for num, center in enumerate(centroids):
			diff = torch.sqrt(torch.sum((data-center)**2))	
			if diff <= radii[num]+args.epsilon:
				count_expand += 1
				break

	risk_test = count/float(n_test)
	advRisk_test = count_expand/float(n_test)
	print('Risk for test data:', '{:.2%}'.format(risk_test))
	print('Adversarial risk for test data:', '{:.2%}'.format(advRisk_test))
	print('')

	#### save the results
	print(args.clusters, '{:.2%}'.format(risk_train), 
		'{:.2%}'.format(risk_test), '{:.2%}'.format(advRisk_train), 
		'{:.2%}'.format(advRisk_test), file=res_file)
	res_file.flush()
