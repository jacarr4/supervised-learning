My code can be accessed at: https://github.gatech.edu/jcarr61/supervised-learning.git
There are several required python packages. I have included a virtual environment containing them all:
* . mp1/bin/activate
This is not required if your environment contains the packages already.

The script run_learner.py contains the code. It can be run as follows:
* python -m run_learner --learner <learner> --dataset <dataset> --kernel <kernel>
Learner can be one of:
* decision_tree
* neural_network
* boosting
* svm
* knn
Dataset can be one of:
* pima
* digits
Kernel can be one of (others are possible but only these appear in my report):
* linear
* rbf
The kernel is only used when --learner is svm.

run_learner.py will show the hyperparameter optimization graphs on the screen, followed by the learning/scalability/performance curves.

That is all. Thank you!

Jacob Carr