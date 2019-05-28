#### problem: empirically estimate the intrinsic robustness on real dataset
####          (l-infinity norm-bounded perturbations on MNIST and CIFAR-10 datasets)
####
#### strategy: cover the top-q densest images with union of rectangles as the robust region,
####           then expand them by epsilon to obtain the error region (rects-complement).
####

import load_data as ld
import numpy as np 
import random
from pathlib import Path
import setproctitle
import os

if __name__ == "__main__":
	args = ld.argparser(dataset='mnist', metric='infinity', epsilon=0.3,
						k=50, q=0.703, clusters=10, iter=30, repeat=10)
	# args = ld.argparser(dataset='cifar', metric='infinity', epsilon=0.007843,
	# 				k=50, q=0.683, clusters=10, iter=30, repeat=10)
	setproctitle.setproctitle('python')

	print('dataset: {dataset}\t\t' 'metric: {metric}\t\t' 'epsilon: {epsilon}\t\t' 'k: {k}\t\t' 
			'q: {q}\t\t' '#clusters: {clusters}\t\t' '#iter: {iter}\t\t' '#repeat: {repeat}'.format(
			dataset=args.dataset, metric=args.metric, epsilon=args.epsilon, k=args.k,
			q=args.q, clusters=args.clusters, iter=args.iter, repeat=args.repeat))

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

	#### get the top q-quantile densest images 
	filepath = './eda/knn_'+args.dataset+'_'+args.metric+'_'+str(args.k)+'.txt'
	if Path(filepath).exists():
		radii = np.loadtxt(filepath, dtype=float)
	else:
		raise ValueError('knn distance estimates file not exist')

	n_images = int(np.floor(len(train_data)*args.q))
	indices = radii.argsort()[:n_images]
	train_dense = train_data[indices,:]
	# print('sorted l1 distance to the k-th nearest neighbour: ', radii[indices])

	#### perform K-means clustering on the selected dense images using L1 metric
	print('=============== perform kmeans clustering (clusters = ' + str(args.clusters) + ') ===============')
	risk_train = np.zeros(args.repeat)
	risk_test = np.zeros(args.repeat)
	advRisk_train = np.zeros(args.repeat)
	advRisk_test = np.zeros(args.repeat)

	for j in range(args.repeat):
		print('=============== Experiment No.'+str(j)+ ' ===============')
		labels = np.zeros(train_dense.shape[0])
		label_sum = 0.0
		centroid_index = random.sample(range(1, train_dense.shape[0]), args.clusters)
		centroid  =  train_dense[centroid_index]		# record the center of the rectangles
		edge_size = np.zeros((args.clusters, train_data.shape[1]))		# record the edge of the rectangles

		for iterations in range(args.iter):
			print("Iteration " + str(iterations))
			# partition the data
			for idx, data in enumerate(train_dense):
				dist = []
				for center in centroid:
					dist.append(np.linalg.norm(data-center, ord=1))		# compute the l1 distance			
				labels[idx] = dist.index(min(dist))

			# update clusters 
			for cluster_number in range(args.clusters):
				index = []
				for idx, label in enumerate(labels):
					if label == cluster_number:
						index.append(idx)
				temp_data = train_dense[index]
				print('Updating cluster ' + str(cluster_number))
				centroid[cluster_number] = np.mean(temp_data, axis=0)
				temp_dist = np.linalg.norm(temp_data-centroid[cluster_number], axis=0, ord=np.inf)
				edge_size[cluster_number,:] = temp_dist
				
			# label sum not changed -> iterations converge
			print(' * label sum:', np.sum(labels))
			if label_sum == np.sum(labels):
				print('kmeans algorithm converges!')
				break
			else:
				label_sum = np.sum(labels)

		#### measure the expansion on training data
		count = 0
		count_expand = 0

		for idx, data in enumerate(train_data):
			for num, center in enumerate(centroid):
				diff = np.abs(data-center)		
				if np.max(diff-edge_size[num,:]) <= 0:
					count += 1
					break
			for num, center in enumerate(centroid):
				diff = np.abs(data-center)
				if np.max(diff-edge_size[num,:]) <= args.epsilon:
					count_expand += 1
					break

		risk_train[j] = (len(train_data)-count_expand) / len(train_data)
		advRisk_train[j] = (len(train_data)-count) / len(train_data)
		print('')
		print('Risk for train data:', '{:.2%}'.format(risk_train[j]))
		print('Adversarial risk for train data:', '{:.2%}'.format(advRisk_train[j]))

		#### measure the generalization on testing data
		count= 0
		count_expand = 0

		for idx, data in enumerate(test_data):
			for num, center in enumerate(centroid):
				diff = np.abs(data-center)
				if np.max(diff-edge_size[num,:]) <= 0:
					count += 1
					break
			for num, center in enumerate(centroid):
				diff = np.abs(data-center)
				if np.max(diff-edge_size[num,:]) <= args.epsilon:
					count_expand += 1
					break

		risk_test[j] = (len(test_data)-count_expand) / len(test_data)
		advRisk_test[j] = (len(test_data)-count) / len(test_data)
		print('Risk for test data:', '{:.2%}'.format(risk_test[j]))
		print('Adversarial risk for test data:', '{:.2%}'.format(advRisk_test[j]))
		print('')

	#### save the results
	res_file = open(res_filepath, "w")
	print(args.clusters, '{:.3}'.format(args.q), '{:.2%}'.format(np.mean(risk_train)), 
		'({:.2%})'.format(np.std(risk_train)), '{:.2%}'.format(np.mean(risk_test)), 
		'({:.2%})'.format(np.std(risk_test)), '{:.2%}'.format(np.mean(advRisk_train)), 
		'({:.2%})'.format(np.std(advRisk_train)), '{:.2%}'.format(np.mean(advRisk_test)), 
		'({:.2%})'.format(np.std(advRisk_test)), file=res_file)

