import argparse
from   enum                    import IntEnum
import graphviz
import matplotlib.pyplot       as plt
import numpy                   as np
import pandas                  as pd
from   sklearn                 import datasets
from   sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier
from   sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
from   sklearn.neighbors       import KNeighborsClassifier
from   sklearn.neural_network  import MLPClassifier
from   sklearn.svm             import SVC
from   sklearn.tree            import export_graphviz, DecisionTreeClassifier, plot_tree

class Dataset( IntEnum ):
    PimaIndians = 0
    Digits = 1

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

class ClassifierGroup:
    def __init__( self, dataset : Dataset ):
        if dataset == Dataset.PimaIndians:
            self.get_pima_indians_data()
            self.ravel = True
        elif dataset == Dataset.Digits:
            self.get_digits_data()
            self.ravel = False
        else:
            raise ValueError( "Dataset %s not supported" % dataset )

        if self.ravel:
            self.y_train = self.y_train.values.ravel()
            self.y_full_training_set = self.y_full_training_set.values.ravel()
            self.y_full_testing_set = self.y_full_testing_set.values.ravel()
            self.y_validate = self.y_validate.values.ravel()

    def get_pima_indians_data( self ):
        dtf = pd.read_csv('pima/pima-indians-diabetes.csv')
        dtf.columns = [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome' ]

        data = dtf[ [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ] ]
        target = dtf[ [ 'Outcome' ] ]

        X_train, X_test, y_train, y_test = train_test_split( data, target, test_size = 0.2, random_state = 0 )
        self.X_full_training_set = X_train.copy()
        self.X_full_testing_set = X_test.copy()
        self.y_full_training_set = y_train.copy()
        self.y_full_testing_set = y_test.copy()

        X_train, X_validate, y_train, y_validate = train_test_split( X_train, y_train, test_size = 0.2, random_state = 0 )
        self.X_train = X_train.copy()
        self.X_validate = X_validate.copy()
        self.y_train = y_train.copy()
        self.y_validate = y_validate.copy()

    def get_digits_data( self ):
        digits = datasets.load_digits()

        X_train, X_test, y_train, y_test = train_test_split( digits.data, digits.target, test_size = 0.2, random_state = 0 )
        self.X_full_training_set = X_train.copy()
        self.X_full_testing_set = X_test.copy()
        self.y_full_training_set = y_train.copy()
        self.y_full_testing_set = y_test.copy()

        X_train, X_validate, y_train, y_validate = train_test_split( X_train, y_train, test_size = 0.2, random_state = 0 )
        self.X_train = X_train.copy()
        self.X_validate = X_validate.copy()
        self.y_train = y_train.copy()
        self.y_validate = y_validate.copy()
    
    def get_test_score( self, clf ):
        clf.fit( self.X_full_training_set, self.y_full_training_set )
        return clf, clf.score( self.X_full_testing_set, self.y_full_testing_set )
    
    def get_validation_score( self, clf ):
        clf.fit( self.X_train, self.y_train )
        return clf.score( self.X_validate, self.y_validate )
    
    def get_score( self, clf ):
        clf.fit( self.X_full_training_set, self.y_full_training_set )
        return clf.score( self.X_full_testing_set, self.y_full_testing_set )
    
    def get_score_and_classifier( self, clf ):
        clf.fit( self.X_train, self.y_train )
        return clf, clf.score( self.X_test, self.y_test )
    
    def get_training_score( self, clf ):
        clf.fit( self.X_train, self.y_train )
        return clf.score( self.X_train, self.y_train )
    
    def decision_tree( self ):
        clf = DecisionTreeClassifier()
        clf, score = self.get_score_and_classifier( clf )
        return score
    
    def decision_tree_score( self ):
        clf = DecisionTreeClassifier()
        clf, score = self.get_test_score( clf )
        return score 
    
    def neural_network( self ):
        clf = MLPClassifier()
        return self.get_score( clf )
    
    def boosting( self ):
        clf = AdaBoostClassifier()
        return self.get_score( clf )
    
    def boosting2( self ):
        clf = GradientBoostingClassifier()
        return self.get_score( clf )
    
    def svm( self ):
        clf = SVC( kernel = 'poly' )
        return self.get_score( clf )
    
    def knn( self ):
        clf = KNeighborsClassifier()
        return self.get_score( clf )
    
    def find_best_parameter( self, classifier_ctor, param_name, possible_values ):
        best_score = 0
        best_value = None
        best_clf = None
        for v in possible_values:
            clf = classifier_ctor( **{ param_name: v } )
            new_score = self.get_training_score( clf )
            if new_score > best_score:
                best_score = new_score
                best_value = v
                best_clf = clf
            print( f'{param_name}={v}: {new_score}' )
        print( f'Best score: {best_score} was achieved with parameter {param_name}={best_value}' )
        return best_value
    
    def find_best_hyperparameter( self, classifier_ctor, param_name, possible_values, categorical = False ):
        best_validation_score = 0
        best_value = None
        best_clf = None
        scores = []
        for v in possible_values:
            print( { param_name : v } )
            clf = classifier_ctor( **{ param_name: v } )
            new_score = self.get_validation_score( clf )
            scores.append( new_score )
            if new_score > best_validation_score:
                best_validation_score = new_score
                best_value = v
                best_clf = clf
            # print( f'{param_name}={v}: {new_score}' )
        print( f'Best score: {best_validation_score} was achieved with hyperparameter {param_name}={best_value}' )
        if categorical:
            plt.plot( possible_values, scores, 'o' )
        else:
            plt.plot( possible_values, scores )
        plt.xlabel( f'{param_name}' )
        plt.ylabel( 'Validation Score' )
        plt.suptitle( f'{param_name}' )
        plt.show()
        return best_value, best_clf

def make_plot( fig, axes, clf, X, y ):
    title = "Learning Curves"
    cv = ShuffleSplit( n_splits = 100, test_size = 0.2, random_state = 0 )
    plot_learning_curve( clf, title, X, y, axes = axes, ylim=( 0.2, 1.01 ), cv = cv, n_jobs = 4 )

def optimize_hyperparams( name, classifier_ctor, dataset, params ):
    hyperparams = {}
    c = ClassifierGroup( dataset )
    print( f'Dataset: {dataset.name}' )
    clf = classifier_ctor()
    print( '%s training score with no hyperparameter optimization: %.3f' % ( name, c.get_training_score( clf ) ) )
    print( '%s test score with no hyperparameter optimization: %.3f' % ( name, c.get_test_score( clf )[ 1 ] ) )

    for param_name, ( possible_values, is_categorical ) in params.items():
        hyperparams[ param_name ], clf = c.find_best_hyperparameter( classifier_ctor, param_name, possible_values, categorical = is_categorical )
    
    print( hyperparams )
    
    print( '%s training score with hyperparameter optimization: %.3f' % ( name, c.get_training_score( clf ) ) )
    print( '%s test score with hyperparameter optimization: %.3f' % ( name, c.get_test_score( clf )[ 1 ] ) )
    return hyperparams

def optimize_decision_tree_hyperparams( dataset ):
    params = { 'criterion': ( [ 'gini', 'entropy' ], True ),
               'splitter':  ( [ 'best', 'random' ], True ),
               'max_depth': ( [ i for i in range( 1, 64 ) ], False ) }
    return optimize_hyperparams( 'Decision Tree', DecisionTreeClassifier, dataset, params )

def optimize_neural_network_hyperparams( dataset ):
    params = { 'activation': ( [ 'identity', 'logistic', 'tanh', 'relu' ], True ),
               'solver':     ( [ 'lbfgs', 'sgd', 'adam' ],                 True ),
               'max_iter':   ( [ i for i in range( 1, 200 ) ],             False ) }
    return optimize_hyperparams( 'Neural Network', MLPClassifier, dataset, params )

def optimize_boosting_hyperparams( dataset ):
    params = { 'learning_rate': ( [ 0.01 * i for i in range( 1, 20 ) ], False ),
               'n_estimators':  ( [ i for i in range( 1, 100 ) ], False ),
               'subsample':     ( [ 0.1 * i for i in range( 1, 10 ) ], False ) }
    return optimize_hyperparams( 'Boosting', GradientBoostingClassifier, dataset, params )

def optimize_svm_hyperparams( dataset, kernel ):
    if kernel is None:
        raise ValueError( 'Invalid kernel' )
    params = { 'C': ( [ 0.1 * i for i in range( 1, 20 ) ], False ),
               'gamma': ( [ 0.01 * i for i in range( 1, 100 ) ], False ),
               'kernel': ( [ kernel ], True ) }
    return optimize_hyperparams( 'Support Vector Machine', SVC, dataset, params )

def optimize_knn_hyperparams( dataset ):
    params = { 'n_neighbors': ( [ i for i in range( 1, 20 ) ], False ),
               'weights':     ( [ 'uniform', 'distance' ], True ),
               'algorithm':   ( [ 'auto', 'ball_tree', 'kd_tree', 'brute' ], True ) }
    return optimize_hyperparams( 'K-Nearest Neighbors', KNeighborsClassifier, dataset, params )

def load_pima():
    dtf = pd.read_csv('pima/pima-indians-diabetes.csv')
    dtf.columns = [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome' ]

    data = dtf[ [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ] ]
    target = dtf[ [ 'Outcome' ] ]

    return data, target

def load_dataset( dataset ):
    if dataset == Dataset.Digits:
        return datasets.load_digits( return_X_y = True )
    else:
        return  load_pima()

def neural_network_learning_graphs( dataset, params ):
    fig, axes = plt.subplots( 3, 1, figsize = ( 10, 15 ) )
    X, y = load_dataset( dataset )
    y = y.values.ravel() if dataset == Dataset.PimaIndians else y
    print( params )
    clf = MLPClassifier( **params )
    make_plot( fig, axes, clf, X, y )
    plt.show()

def compute_learning_graphs( classifier_ctor, dataset, params ):
    fig, axes = plt.subplots( 3, 1, figsize = ( 10, 15 ) )
    X, y = load_dataset( dataset )
    y = y.values.ravel() if dataset == Dataset.PimaIndians else y
    print( params )
    clf = classifier_ctor( **params )
    make_plot( fig, axes, clf, X, y )
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--learner', action = 'store', dest = 'learner', required = True )
    parser.add_argument( '--dataset', action = 'store', dest = 'dataset', required = True )
    parser.add_argument( '--kernel', action = 'store', dest = 'kernel', required = False )
    args = parser.parse_args()

    if args.dataset == 'pima':
        dataset = Dataset.PimaIndians
    else:
        dataset = Dataset.Digits

    if args.learner == 'decision_tree':
        params = optimize_decision_tree_hyperparams( dataset )
        compute_learning_graphs( DecisionTreeClassifier, dataset, params )
    elif args.learner == 'neural_network':
        params = optimize_neural_network_hyperparams( dataset )
        compute_learning_graphs( MLPClassifier, dataset, params )
    elif args.learner == 'boosting':
        params = optimize_boosting_hyperparams( dataset )
        compute_learning_graphs( GradientBoostingClassifier, dataset, params )
    elif args.learner == 'svm':
        params = optimize_svm_hyperparams( dataset, args.kernel )
        compute_learning_graphs( SVC, dataset, params )
    elif args.learner == 'knn':
        params = optimize_knn_hyperparams( dataset )
        compute_learning_graphs( KNeighborsClassifier, dataset, params )
    else:
        raise ValueError( 'Invalid learner argument' )
