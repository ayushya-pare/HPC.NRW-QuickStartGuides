"""Use linear SVM for iris classification."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def train_test_svc(
    xtrain: np.ndarray, xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray
) -> Tuple[LinearSVC, float]:
    """Train a linear SVM classifier and compute the accuracy on the test set.

    Define a linear SVM classifier. Train it and compute the accuracy on the test set.
    Express the accuracy as a percentage and round to one decimal place.

    Args:
        xtrain (np.ndarray): The training data.
        xtest (np.ndarray): The test data.
        ytrain (np.ndarray): The training labels.
        ytest (np.ndarray): The test labels.

    Returns:
        Tuple[LinearSVC, float]: The trained linear SVM classifier and the accuracy of the model on the test set.
    """
    # 4.1. create and train linear SVM
    # TODO
    clf = LinearSVC()
    clf.fit(xtrain, ytrain)

    # 4.2. predict on test set and calculate accuracy
    # TODO
    y_pred = clf.predict(xtest)
    accuracy = np.mean(y_pred == ytest) * 100
    #accuracy = round(accuracy, 1)
    return clf, accuracy


if __name__ == "__main__":
    # 1. load the Iris dataset
    # TODO
    iris = load_iris()

    # 2. get access to data, labels and class names
    # TODO
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # 3. split data into training and test sets
    # TODO
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. use function train_test_svc to train a linear SVM model and calculate accuracy on test set
    # TODO
    model, accuracy = train_test_svc(X_train, X_test, y_train, y_test)

    # 5. print accuracy
    # TODO
    print(f"Test accuracy: {accuracy}%")

    # 6. plot confusion matrix
    # TODO
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names)
    plt.show()

# ==============================
# ==============================
# ==============================

"""Use soft-margin SVM for face recognition."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def plot_image_matrix(images, titles, h, w, n_row=3, n_col=4) -> None:
    """Plot a matrix of images.

    Args:
        images (np.ndarray): The array of the images.
        titles (np.ndarray or list): The titles of the images.
        h (int): The height of one image.
        w (int): The width of one image.
        n_row (int): The number of rows of images to plot.
        n_col (int): The number of columns of images to plot.
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    indices = np.arange(n_row * n_col)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[indices[i]].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[indices[i]], size=12)
        plt.xticks(())
        plt.yticks(())


def cv_svm(xtrain: np.ndarray, ytrain: np.ndarray) -> GridSearchCV:
    """Train and cross-validate a soft-margin SVM classifier with the grid search.

    Define an SVM classifier. Use the grid search and a 5-fold cross-validation
    to find the best value for the hyperparameters 'C' and 'kernel'.

    Args:
        xtrain (np.ndarray): The training data.
        ytrain (np.ndarray): The training labels.

    Returns:
        GridSearchCV: The trained model that was cross-validated with the grid search.
    """
    # 6.1. define dictionary with parameter grids
    # TODO
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }

    # 6.2. initialize svm classifier and perform grid search
    # TODO
    svc = svm.SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=5)
    grid_search.fit(xtrain, ytrain)
    return grid_search


if __name__ == "__main__":
    # 1. load dataset 'Labeled Faces in the Wild';
    # take only classes with at least 70 images; downsize images for speed up
    # TODO
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    h = lfw_people.images.shape[1]
    w = lfw_people.images.shape[2]

    # 2. gather information about the dataset
    # print number of samples, number of image features (pixels) and number of classes
    # TODO
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of image features (pixels): {X.shape[1]}")
    print(f"Number of classes: {len(target_names)}")

    # 3. plot some images of dataset
    # TODO
    plot_image_matrix(lfw_people.images, lfw_people.target, h, w)
    plt.show()

    # 4. split data into training and test data
    # TODO
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. use 'StandardScaler' on train data and scale both train and test data
    # TODO
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. use function 'cv_svm' to perform hyperparameter search with cross validation
    # TODO
    model = cv_svm(X_train_scaled, y_train)

    # 7. print parameters found with cross-validation
    # TODO
    print("Best parameters found:", model.best_params_)

    # 8. compute and print accuracy of best model on test set
    # TODO
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # 9. Plot images together with predicitons
    # TODO
    plot_image_matrix(X_test, y_pred, h, w)
    plt.show()

    # (optional) 10. calculate and plot ROC curve and AUC
    # TODO
    from sklearn.metrics import roc_curve, auc
    n_classes = len(np.unique(y_test))
    y_score = model.decision_function(X_test_scaled)
    plt.figure()
    chosen_class = 2
    # Make binary labels: 1 for chosen class, 0 for all others
    y_test_binary = (y_test == chosen_class).astype(int)
    
    # Get scores for class i
    scores_for_chosen_class = y_score[:, 2]
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test_binary, scores_for_chosen_class)
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    print(f"Class {chosen_class} AUC: {roc_auc:.2f}")
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'Class {chosen_class} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

# ==============================
# ==============================
# ==============================

"""Analyse corona virus case data with support vector machines."""

import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def plot_curve(curve: np.ndarray) -> None:
    """Plot a curve of confirmed coronavirus cases.

    Args:
        curve (np.ndarray): y-axis of the number of covid cases for len(curve) days.
    """
    plt.figure(figsize=(10, 5))
    plt.title("Number of confirmed Coronavirus cases in the world", size=20)
    plt.plot(range(len(curve)), curve)


def plot_prediction(
    raw_data: np.ndarray,
    available_days: int,
    max_value: np.int64,
    recursive_predictions: np.ndarray,
) -> None:
    """Plot the raw data and the predicted values.

    Args:
        raw_data (np.ndarray): The raw data to plot.
        available_days (int): The days to plot.
        max_value (np.int64): The maximum number of covid cases.
        recursive_predictions (np.ndarray): Predictions for the following days.
    """
    day1 = datetime.datetime.strptime("1/22/2020", "%m/%d/%Y")
    day50 = day1 + datetime.timedelta(days=50)
    available_date_range = [
        day50 + datetime.timedelta(days=x) for x in range(available_days)
    ]
    future_date_range = [
        available_date_range[-1] + datetime.timedelta(days=x)
        for x in range(1, len(recursive_predictions) + 1)
    ]
    full_available_date_range = [
        day1 + datetime.timedelta(days=x) for x in range(len(raw_data))
    ]

    plt.figure(figsize=(20, 10))
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.plot(full_available_date_range, raw_data)
    plt.plot(future_date_range, recursive_predictions, linestyle="dashed", color="red")
    plt.legend(["Real values", "Predictions"], prop={"size": 20}, loc=2)
    plt.xticks(rotation=90)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, max_value * 2))

    plt.savefig("./time_series_preds.png")

    print("Number of confirmed cases:")
    for i, day in enumerate(future_date_range):
        print(day.date(), ":", recursive_predictions[i])


def recursive_forecast(
    model: SVR, start_values: np.ndarray, days_to_forecast: int
) -> np.ndarray:
    """Recursivley use new predictions to generate time series predictions for the future.

    Args:
        model (SVR): The trained model for future predictions.
        start_values (np.ndarray): Array of shape (numberOfPreviousData,).
        days_to_forecast (int): Number of days to forecast.

    Returns:
        np.ndarray: Predictions for the next days_to_forecast.
    """
    moving_x = start_values.copy()
    predictions = np.zeros(days_to_forecast)
    for i in range(days_to_forecast):
        mov_res = moving_x.reshape(1, -1)
        # 7.1. predict values for moving x
        new_forecast = model.predict(mov_res)
        # 7.2. shift window and add new prediction
        moving_x[:-1] = moving_x[1:]
        moving_x[-1] = new_forecast[0]
        predictions[i] = new_forecast[0]
    return predictions


def cv_svr(train_x: np.ndarray, train_y: np.ndarray) -> GridSearchCV:
    """Find the best parameters for a SVR model with grid search.

    Train and cross-validate a soft margin SVM regressor with the grid search.

    Define an SVM regressor. Use the grid search and a 5-fold cross-validation
    to find the best value for the hyperparameters 'C', 'gamma' and 'epsilon'.

    Args:
        train_x (np.ndarray): The training data.
        train_y (np.ndarray): The training labels.

    Returns:
        GridSearchCV: The trained model that was cross-validated with the grid search.
    """
    # 5.1. define dictionary with parameter grids
    # TODO
    dict_parameters = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "epsilon": [0.001, 0.01, 0.1],
    }
    # 5.2. initialize svm regressor and perform grid search
    # TODO
    svm_regressor = SVR()
    svm_grid_search = GridSearchCV(
        estimator=svm_regressor,
        param_grid=dict_parameters,
        cv=5
    )

    return svm_grid_search


if __name__ == "__main__":
    # read dataset
    df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
        "master/csse_covid_19_data/csse_covid_19_time_series/"
        "time_series_covid19_confirmed_global.csv"
    )

    # take a look at the dataframe with print(df.head()) to see how the data is structured.
    # head() will return the first 5 rows of the dataframe.

    # only choose first 78 days.
    df = df[df.columns[:82]]

    # compute total number of cases
    first_date_index = df.columns.get_loc("1/22/20")
    df = df[df.columns[first_date_index:]]
    raw_data = df.to_numpy()
    raw_data = np.sum(raw_data, axis=0)

    # only use last part of data
    raw_data_short = raw_data[50:]

    # 1. plot curves using 'plot_curve' function above
    # TODO
    plot_curve(raw_data)
    plot_curve(raw_data_short)
    plt.show()

    # 2. set number of days you want to forecast
    # and number of days that will be taken into account for forecast
    days_to_forecast = 5  # TODO
    num_previous_data = 7  # TODO

    # 3. build dataset for training and testing
    num_train_test_records = len(raw_data_short) - num_previous_data
    x = np.zeros((num_train_test_records, num_previous_data))
    y = np.zeros(num_train_test_records)
    for i in range(num_train_test_records):
        for j in range(num_previous_data):
                x[i, j] = raw_data_short[i + j]  # Use 7 consecutive days as features
        y[i] = raw_data_short[i + num_previous_data]  # Predict the next day after the window

    # split dataset into train and test sets
    x_train = x.copy()
    y_train = y.copy()
    #x_test = raw_data_short[-num_previous_data:]
    x_test = raw_data_short[len(raw_data_short) - num_previous_data :]

    # 4. normalize input data to its max value, such that it lies between [0,1]
    # TODO
    x_train = x_train / np.max(x_train)
    y_train = y_train / np.max(y_train)
    x_test = x_test / np.max(x_test)

    # 5. use function 'cv_svr' to perform hyperparameter search with cross validation
    # TODO
    svm_model = cv_svr(x_train, y_train)
    svm_model.fit(x_train, y_train)

    # 6. print parameters found with cross-validation
    # TODO
    print("Best parameters found: ", svm_model.best_params_)

    # 8. make predictions for next 5 days; round and denormalize predictions
    # TODO
    predictions = recursive_forecast(svm_model, x_test, days_to_forecast)
    predictions = np.round(predictions * np.max(raw_data_short))

    # 9. use 'plot_prediction' to plot predicted results
    # TODO
    plot_prediction(raw_data_short,
                    days_to_forecast, 
                    np.max(raw_data_short), 
                    predictions)
    plt.show()

