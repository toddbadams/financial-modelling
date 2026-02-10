from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

if TYPE_CHECKING:
    from test_data import TestData


@dataclass(frozen=True)
class BaselineTrainResults:
    label: str
    accuracy: float
    f1_macro: float


class BaselineTrain:

    def run(self, test_data: "TestData") -> list[BaselineTrainResults]:
        results: list[BaselineTrainResults] = []

        for label_col in test_data.label_cols:
            X_train = test_data.train_features
            y_train = test_data.train_lables[label_col]
            X_test = test_data.test_features
            y_test = test_data.test_lables[label_col]

            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,
                n_jobs=-1,
                eval_metric="mlogloss",
                seed=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")

            results.append(
                BaselineTrainResults(
                    label=label_col, accuracy=float(acc), f1_macro=float(f1)
                )
            )

        results = sorted(results, key=lambda r: r.accuracy)
        return results

    def hyper_param_tunning(
        self, test_data: "TestData", results: list[BaselineTrainResults]
    ):

        best_label = max(results, key=lambda x: x.f1_macro).label
        # print("Best label:", best_label)

        # best_label = "label_la6_th0.030"  # Replace with custom label from results

        X_train = test_data.train_features
        y_train = test_data.train[best_label]
        X_test = test_data.test_features
        y_test = test_data.test[best_label]

        param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        tscv = TimeSeriesSplit(n_splits=5)

        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_jobs=-1,
            eval_metric="mlogloss",
            seed=42,
        )

        grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=tscv,
            scoring="accuracy",
            verbose=1,
            n_jobs=-1,
        )

        grid.fit(X_train, y_train)
        print("Best params:", grid.best_params_)
        print("Best CV accuracy:", grid.best_score_)

        best_model: XGBClassifier = grid.best_estimator_
        test_preds = best_model.predict(X_test)
        print(classification_report(y_test, test_preds))
