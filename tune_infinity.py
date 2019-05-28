#### l2-norm bounded perturbations
#### tune the number of clusters for the given epsilon using binary search over q

import load_data as ld
import numpy as np 

import random
from pathlib import Path
import setproctitle
import os

if __name__ == "__main__":
	args = ld.argparser(dataset='mnist', metric='infinity', epsilon=0.3, 
						alpha=0.01, k=50, iter=30)
	# args = ld.argparser(dataset='cifar', metric='infinity', epsilon=0.007843, 
	# 					alpha=0.05, k=50, iter=30)
	setproctitle.setproctitle('python')

	print('dataset: {dataset}\t\t' 'metric: {metric}\t\t' 'epsilon: {epsilon}\t\t' 
			'alpha: {alpha}\t\t' 'k: {k}\t\t' '#iter: {iter}\t\t'.format(
			dataset=args.dataset, metric=args.metric, epsilon=args.epsilon, 
			alpha=args.alpha, k=args.k, iter=args.iter))

	#### define results path
	res_filepath = (os.path.dirname('./results/coarse_tune/'+args.proctitle)+'/alpha_'+str(args.alpha))
	print("saving file to {}".format(res_filepath+'/epsilon_'+str(args.epsilon)))
	if not os.path.exists(res_filepath):
		os.makedirs(res_filepath)

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

	#### get the top q quantile densest images 
	filepath = './eda/knn_'+args.dataset+'_'+args.metric+'_'+str(args.k)+'.txt'
	if Path(filepath).exists():
		radii = np.loadtxt(filepath, dtype=float)
	else:
		raise ValueError('knn distance estimates file not exist')

	#### kmeans clustering for each combination (n_cluster, q)
	cluster_arr = np.arange(5, 11, 5)
	res_file = open(res_filepath+'/epsilon_'+str(args.epsilon)+'.txt', "w")
	best_file = open(res_filepath+'/epsilon_'+str(args.epsilon)+'_best.txt', "w")

	for n_cluster in cluster_arr:
		flag = False
		advRisk_test_best = 1.0

		# binary search for best quantile in (0,1)
		q_lower = 0.0
		q_upper = 1.0
		while q_upper-q_lower > 0.005:	
			q = (q_lower+q_upper)/2
			n_images = int(np.floor(len(train_data)*q))
			indices = radii.argsort()[:n_images]
			train_dense = train_data[indices,:]
			# print('sorted l1 distance to the k-th nearest neighbour: ', radii[indices])

			#### perform K-means clustering on the selected dense images using L1 metric
			print('=============== #clusters='+str(n_cluster)+', quantile='+str(q)+' ===============')
			labels = np.zeros(train_dense.shape[0])
			label_sum = 0.0
			centroid_index = random.sample(range(1, train_dense.shape[0]), n_cluster)
			centroid  =  train_dense[centroid_index]		# record the center of the rectangles
			edge_size = np.zeros((n_cluster, train_data.shape[1]))		# record the edge of the rectangles

			for iterations in range(args.iter):
				print("Iteration " + str(iterations))
				# partition the data
				for idx, data in enumerate(train_dense):
					dist = []
					for center in centroid:
						dist.append(np.linalg.norm(data-center, ord=1))		# compute the l1 distance
					labels[idx] = dist.index(min(dist))

				# update clusters 
				for cluster_number in range(n_cluster):
					index = []
					for idx, label in enumerate(labels):
						if label == cluster_number:
							index.append(idx)
					temp_data = train_dense[index]
					print('Updating cluster ' + str(cluster_number))
					centroid[cluster_number] = np.mean(temp_data, axis=0)
					temp_dist = np.linalg.norm(temp_data-centroid[cluster_number], axis=0, ord=np.inf)
					edge_size[cluster_number,:] = temp_dist

				# label sum not changed -> iteration converge
				print(' * label sum:',np.sum(labels))
				if label_sum == np.sum(labels):
					print('kmeans algorithm converges!')
					break
				else:
					label_sum = np.sum(labels)

			print('')
			print('=============== results on '+args.dataset+
					' with (L-'+args.metric+', '+str(args.epsilon)+') ===============')
			print('total number of rectangles available:', n_cluster)
			print('initial quantile of densest image to cover:', '{:.2%}'.format(q))

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

			risk_train = (len(train_data)-count_expand) / len(train_data)
			advRisk_train = (len(train_data)-count) / len(train_data)
			print('Risk for train data:', '{:.2%}'.format(risk_train))
			print('Adversarial risk for train data:', '{:.2%}'.format(advRisk_train))

			#### test the generalization on testing data
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

			risk_test = (len(test_data)-count_expand) / len(test_data)
			advRisk_test = (len(test_data)-count) / len(test_data)
			print('Risk for test data:', '{:.2%}'.format(risk_test))
			print('Adversarial risk for test data:', '{:.2%}'.format(advRisk_test))
			print('')

			print('{}\t\t'.format(n_cluster), '{:.3}\t\t'.format(q), 
				  '{:.2%}\t\t'.format(risk_train), '{:.2%}\t\t'.format(risk_test), 
				  '{:.2%}\t\t'.format(advRisk_train), '{:.2%}'.format(advRisk_test), file=res_file)
			res_file.flush()

			if risk_train >= args.alpha and risk_test >= args.alpha:
				flag = True
				q_lower = q 		# measure of error region > alpha -> increasing q
				
				if advRisk_test <= advRisk_test_best:
					q_best = q
					risk_train_best = risk_train
					advRisk_train_best = advRisk_train
					risk_test_best = risk_test
					advRisk_test_best = advRisk_test
			else:
				q_upper = q 		# measure of error region too small -> decreasing q
				print('measure of error region is too small.')

		# save the best results for each #clusters
		if flag:
			print('{}\t\t'.format(n_cluster), '{:.2%}\t\t'.format(q_best), 
				  '{:.2%}\t\t'.format(risk_train_best), '{:.2%}\t\t'.format(risk_test_best), 
				  '{:.2%}\t\t'.format(advRisk_train_best), 
				  '{:.2%}'.format(advRisk_test_best), file=best_file)
		else:
			print('{}\t\t'.format(n_cluster), 'N/A', file=best_file)
		best_file.flush()	
