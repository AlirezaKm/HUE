from sklearn.metrics import accuracy_score, roc_auc_score
from ensemble import HashBasedUndersamplingEnsemble
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np


def prepare(X: np.array, y: np.array, minority=None, verbose: bool = False):
    """Preparing Data for Ensemble
    Make the data binary by minority class in the dataset

    :param X: np.array (n_samples, n_features)
        feature matrix

    :param y: np.array (n_samples,)
        label vecotr

    :param minority: int or str (default = None)
    label of minority class
    if you want to set a specific class to be minority class

    :param verbose: bool (default = False)
        verbosity

    :return: np.array, np.array
        X, y returned
    """

    # Get classes and number of them
    classes, counts = np.unique(y, return_counts=True)

    if minority is None:
        # find minority class
        minority = classes[np.argmin(counts)]

    if minority not in classes:
        raise ValueError("class '{}' does not exist".format(
            minority
        ))

    # set new label for data (1 for minority class and -1 for rest of data)
    y_ = np.where(y == minority, 1, -1)

    if verbose:
        information = "[Preparing]\n" \
                      "+ #classes: {}\n" \
                      "+ classes and counts: {}\n" \
                      "+ Minority class: {}\n" \
                      "+ Size of Minority: {}\n" \
                      "+ Size of Majority: {}\n" \
                      "".format(len(classes),
                                list(zip(classes, counts)),
                                minority,
                                np.sum(y_ == 1),
                                np.sum(y_ != 1),
                                )

        print(information)

    return X, y_


def evaluate(
        name,
        base_classifier,
        X,
        y,
        minority_class=None,
        k: int = 5,
        n_runs: int = 20,
        n_iterations: int = 50,
        random_state: int = None,
        verbose: bool = False,
        **kwargs
):
    """Model Evaluation

    :param name: str
        title of this classifier

    :param base_classifier:
        Base Classifier for Hashing-Based Undersampling Ensemble

    :param X: np.array (n_samples, n_features)
        Feature matrix

    :param y: np.array (n_samples,)
        labels vector

    :param minority_class: int or str (default = None)
        label of minority class
        if you want to set a specific class to be minority class

    :param k: int (default = 5)
        number of Folds (KFold)

    :param n_runs: int (default = 20)
        number of runs

    :param n_iterations: int (default = 50)
        number of iterations for Iterative Quantization of Hashing-Based Undersampling Ensemble

    :param random_state: int (default = None)
        seed of random generator

    :param verbose: bool (default = False)
        verbosity

    :return None
    """

    print()
    print("======[Dataset: {}]======".format(
        name
    ))

    np.random.seed(random_state)

    # Output template
    OUTPUT = "[{}] Accuracy: {:.4f}, AUC: {:.4f}"

    # Prepate the data (Make it Binary)
    X, y = prepare(X, y, minority_class, verbose)

    folds = np.zeros((n_runs, 2))
    for run in tqdm(range(n_runs)):

        # Applying k-Fold (k = 5 due to the paper)
        kFold = StratifiedKFold(n_splits=k, shuffle=True)

        # store metrics in this variable
        metrics = np.zeros((k, 2))
        for fold, (trIndexes, tsIndexes) in enumerate(kFold.split(X, y)):
            # Split data to Train and Test
            Xtr, ytr = X[trIndexes], y[trIndexes]
            Xts, yts = X[tsIndexes], y[tsIndexes]

            # Define Model
            model = HashBasedUndersamplingEnsemble(
                base_estimator=base_classifier,
                n_iterations=n_iterations,
                random_state=random_state,
                **kwargs
            )

            # Fit the training data on the model
            model.fit(Xtr, ytr)

            # Predict the test data
            predicted = model.predict(Xts)

            # AUC evaluation
            AUC = roc_auc_score(yts, predicted)

            # Accuracy evaluation
            accuracy = accuracy_score(yts, predicted)

            # Show result for each step
            metrics[fold, :] = [accuracy, AUC]

        folds[run, :] = np.mean(metrics, axis=0)

    print()
    print(OUTPUT.format(
        "Best",
        *np.max(folds, axis=0)
    ))