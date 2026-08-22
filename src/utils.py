import os
import sys
import pandas as pd
import numpy as np
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(
    x_train,
    y_train,
    x_test,
    y_test,
    models,
    param
):
    try:

        report = {}

        for model_name, model in models.items():

            para = param[model_name]

            # Linear Regression has no hyperparameters
            if para:

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )

                gs.fit(x_train, y_train)

                model = gs.best_estimator_

            else:

                model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(
                y_train,
                y_train_pred
            )

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_model_score

            print(
                f"{model_name}: "
                f"Train R2 = {train_model_score:.4f}, "
                f"Test R2 = {test_model_score:.4f}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)