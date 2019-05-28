# Empirically-Mearsuring-Concentration

The goal of this project:
* Connect the adversarial risk to the notion of concentration of measure

* Develop a systematic method to measure the concentration for arbitrary distributions

* Propose an empirical algorithm to measure the concentration for MNIST and CIFAR-10 datasets under L-infinity or L2 metric
    * Step1: Compute the knn distance estimates (k=50) on the given training dataset
    * Step2: Perform Kmeans clustering on the top q-quantile densest images 
    * Step3: Cover each cluster using the corresponding topological subset (rectangle for L-infinity metric, ball for L2 metric)
    * Step4: Obtain the expansion set to get the empirical evaluation of risk and adversarial risk


# Installation & Usage
The code was developed using Python3 on [Anaconda](https://www.anaconda.com/download/#linux)

* Install dependencies:
```text
pip install debacl setproctitle
```

* Examples for measure the empirical concentration based on given datasets:
  ```text
  python main_euclidean.py --dataset mnist --alpha 0.01 --epsilon 1.58 --q 0.5 --clusters 20 
  ```
  ```text
  python binary_search_infinity.py --dataset cifar --alpha 0.05 --epsilon 0.03137
  ```


# What is in this respository?
* ```eda``` folder, including:
  * ```knn_mnist_infinity_50.txt,knn_cifar_infinity_50.txt```: saved knn distance estimates with k=50 on training datasets under L-infinity metric
  * ```knn_mnist_euclidean_50.txt,knn_cifar_euclidean_50.txt```: saved knn distance estimates with k=50 on training datasets under L2 metric

* ```load_data.py```: defines argparser and the dataloaders for MNIST and CIFAR-10 datasets
* ```preliminary.py```: performs knn distance estimation on the given training dataset
* ```main_infinity.py```: main function for emprically measuring the concentration for L-infinity metric based on complement of union of rectangles
* ```main_euclidean.py```: main function for emprically measuring the concentration for L2 metric based on complement of union of balls
* ```binary_search_infinity.py```: implements binary search tunining method (q, clusters) for optimal concentration under L-infinity metric
* ```binary_search_euclidean.py```: implements binary search tuning method (q, clusters) for optimal concentration under L2 metric
