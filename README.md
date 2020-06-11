# Opimization for Machine Learning Mini-Project

### Team : [@olivierlam97](https://github.com/olivierlam97) [@zghonda](https://github.com/zghonda) [@rlaraki](https://github.com/rlaraki)

### Abstract:


### Setup:

Here are the different packages needed to reproduce our experiments:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `math`

To obtain the results : 
Run `./run.py`.

All the following parameters can be tuned in the call of the `full_train_test` method:

- `optimizer` either "BCD" for Block Coordinate Descent or "DFW" for Deep Frank Wolf
- `N_train` the number of samples of the training set
- `N_test` the number of samples of the validation/test set
- `n_iter` the number of iterations
- `n_epochs` the number of epochs
- `batch_size` the batch size used for training, which must be 1 if optimizer = "BCD"
- `d1`, `d2`, `d3` the number of nodes of the 1st, 2nd and 3rd layer
- `alpha`, `gamma`, `rho` the hyper-parameters if optimizer = "BCD"

### Directory structure:

The following directory contrains different text documents, code and data files. The structure is detailed below:

#### Documents:

TODO : put report.tex in report folder

#### Code:

##### Python files:

- `./src/utils/data_utils.py`: contains helper methods for data loading and preprocessing
- `./src/utils/plot_utils.py`: contains helper methods for plotting and to vizualize our results
- `./src/utils/metrics.py`: contains a helper methos to compute mean and standard deviation of an array
- `./src/bcd/ModelBCD.py`: contains a three layers model class with its differents functions to perform training using Block Coordinate Descent optimizer and to compute the test accuracy
- `./src/dfw/baselines/BPGrad.py`: BPGrad optimizer from [https://github.com/oval-group/dfw]
- `./src/dfw/baselines/hinge.py`: MultiClassHingLoss implementation from the same repository as above
- `./src/dfw/dfw.py`: Deep Frank Wolfe optimizer also taken from the same repository
- `./src/dfw/ModelDFW.py` contains a three layers model class with its differents functions, to perform training using Deep Frank Wolfe optimizer and to compute the test accuracy
- `./src/run.py`: main python script which allows to compare the two optimizers and to modify various parameters as described in the #### Setup section









