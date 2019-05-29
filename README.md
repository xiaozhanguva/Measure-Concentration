# Empirically Mearsuring Concentration: Fundamental Limits on Intrinsic Robustness

The goal of this project:
* Develop a systematic framework to measure concentration for arbitrary distributions

* Theoretically, prove that the empirical concentration with respect to special collection of subsets will converge to the actual concentration asymptotically

* Empirically, propose algorithms for measuring concentration of benchmark image distributions under both <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_{\infty}" title="\small \ell_{\infty}" /> and <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_2" title="\small \ell_2" /> distance metrics


# Installation & Usage
The code was developed using Python3 on [Anaconda](https://www.anaconda.com/download/#linux)

* Install Pytorch 0.4.1: 
```text
conda update -n base conda && conda install pytorch=0.4.1 torchvision -c pytorch -y
```

* Install dependencies:
```text
pip install --upgrade pip && pip install scipy sklearn numpy torch setproctitle
```

* Example for empirically measuring concentraion under <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_{\infty}" title="\small \ell_{\infty}" /> metric:

  * First, precompute the distance to the k-th nearest neighours for each training example
    ```text
    python preliminary.py --dataset mnist --metric infinity --k 50 
    ```
  * Next, run the proposed algorithm that finds a robust error region under <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_{\infty}" title="\small \ell_{\infty}" /> 
    ```text
    python main_infinity.py --dataset mnist --epsilon 0.3 --q 0.629 --clusters 10
    ```

* Example for empirically measuring concentraion under <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_2" title="\small \ell_2" /> metric:

  * First, precompute the nearest neighbor indices for each training example
    ```text
    python preliminary.py --dataset cifar --metric euclidean --alpha 0.05
    ```
  * Next, run the proposed algorithm that finds a robust error region under <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_2" title="\small \ell_2" /> 
    ```text
    python main_euclidean.py --dataset cifar --epsilon 0.2453 --alpha 0.05 --clusters 5
    ```

# What is in this respository?
* ```load_data.py```: defines argparser and dataloaders for several benchmark image datasets
* ```preliminary.py```: finds the k-nearest neighbors for each example in a given training dataset
* ```main_infinity.py```: main function for emprically measuring concentration under <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_{\infty}" title="\small \ell_{\infty}" /> metric based on complement of union of hyperrectangles
* ```main_euclidean.py```: main function for emprically measuring concentration under <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_2" title="\small \ell_2" /> metric based on union of balls
* ```tune_infinity.py```: implements the tuning method (gird search for #clusters & binary search for q) for optimal concentration under <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_{\infty}" title="\small \ell_{\infty}" /> metric
* ```tune_euclidean.py```: implements the tuning method (grid search for #clusters) for optimal concentration under <img src="https://latex.codecogs.com/png.latex?\bg_white&space;\small&space;\ell_2" title="\small \ell_2" /> metric
* ```baseline.py```: implements the baseline method that heuristically estimates concentration using linear hyperplane proposed in [Gilmer et al. (2018)](https://arxiv.org/abs/1801.02774)
