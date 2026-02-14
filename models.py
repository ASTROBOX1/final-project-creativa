
from catboost import CatBoostClassifier, CatBoostRegressor
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score,
    precision_score, recall_score
)
import pandas as pd


import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier, XGBRFRegressor
from xgboost import XGBClassifier


classifications_models = {

    # ================= Logistic Regression =================
    "LogisticRegression": LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    # ================= KNN =================
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2
    ),

    # ================= Decision Tree (Simple) =================
    "DecisionTree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),

    # ================= Tuned Decision Tree =================
    # "DecisionTree_Tuned": DecisionTreeClassifier(
    #     criterion="entropy",
    #     max_depth=5,
    #     min_samples_split=10,
    #     min_samples_leaf=5,
    #     min_impurity_decrease=0.01,
    #     ccp_alpha=0.1,
    #     random_state=100
    # ),

    # ================= Random Forest (Simple) =================
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42
    ),

    # # ================= Tuned Random Forest =================
    # "RandomForest_Tuned": RandomForestClassifier(
    #     n_estimators=50,
    #     criterion="entropy",
    #     max_depth=None,
    #     max_features="sqrt",
    #     random_state=100
    # ),

    # ================= Bagging =================
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            min_impurity_decrease=0.01,
            ccp_alpha=0.1,
            random_state=100
        ),
        n_estimators=50,
        bootstrap=True,
        random_state=100
    ),

    # ================= Extra Trees =================
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=50,
        criterion="entropy",
        max_depth=None,
        max_features="sqrt",
        random_state=100
    ),

    # ================= Gradient Boosting =================
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    # ================= AdaBoost =================
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),

    # ================= Naive Bayes =================
    "GaussianNB": GaussianNB(),

    "BernoulliNB": BernoulliNB(
        alpha=1.0,
        binarize=0.0
    ),

    # "MultinomialNB": MultinomialNB(
    #     alpha=1.0,
    #     fit_prior=True
    # ),
    
    "HistGradient Boosting":HistGradientBoostingClassifier(),
    "XGBClassifier":XGBClassifier(),
    "CatBoostClassifier":CatBoostClassifier(),
    "LGBMClassifier":LGBMClassifier()
}
regression_models = {

    "LinearRegression": LinearRegression(
        fit_intercept=True
    ),

    "Ridge": Ridge(
        alpha=1.0,
        solver="auto",
        random_state=42
    ),

    "Lasso": Lasso(
        alpha=0.1,
        max_iter=1000,
        random_state=42
    ),

    "ElasticNet": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=1000,
        random_state=42
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2
    ),

    # "SVR": SVR(
    #     C=1.0,
    #     kernel="rbf",
    #     gamma="scale",
    #     epsilon=0.1
    # ),

    "DecisionTree": DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42
    ),

    "AdaBoost": AdaBoostRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),
    "HistGradient Boosting":HistGradientBoostingRegressor(),
    "XGBRFRegressor":XGBRFRegressor(),
    "CatBoostRegressor":CatBoostRegressor(),
    "LGBMRegressor":LGBMRegressor()
    
}
#classification_models لوب لل 
def run_classification_models(models_dict, X_train, X_test, y_train, y_test):
    results = []

    for name, model in models_dict.items():
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        results.append({
            "model": name,

            # ================= Accuracy =================
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),

            # ================= Precision =================
            "train_precision": precision_score(y_train, y_pred_train, average="weighted"),
            "test_precision": precision_score(y_test, y_pred_test, average="weighted"),

            # ================= Recall =================
            "train_recall": recall_score(y_train, y_pred_train, average="weighted"),
            "test_recall": recall_score(y_test, y_pred_test, average="weighted"),

            # ================= F1 Score =================
            "train_f1": f1_score(y_train, y_pred_train, average="weighted"),
            "test_f1": f1_score(y_test, y_pred_test, average="weighted"),
        })

    return pd.DataFrame(results).sort_values("test_f1", ascending=False)
#linear_model لوب لل 

def run_regression_models(models_dict, X_train, X_test, y_train, y_test):
    results = []

    for name, model in models_dict.items():
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        results.append({
            "model": name,
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test)
        })

    return pd.DataFrame(results).sort_values("test_r2", ascending=False)

