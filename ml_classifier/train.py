

import pandas as pd

from ml_classifier.features import TestData


class Train():

    def run(self, test_data: TestData) :

        
        results = []

        for label_col in [c for c in df_ta.columns if c.startswith("label_")]:
            X_train = test_data.test_features
            X_train, y_train = test_data.train[feature_cols], train_df[label_col]
            X_test,  y_test  = test_df[feature_cols],  test_df[label_col]
            
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=3,
                n_jobs=-1,
                eval_metric='mlogloss',
                seed=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            acc  = accuracy_score(y_test, preds)
            f1   = f1_score(y_test, preds, average='macro')
            
            results.append({
                "label": label_col,
                "accuracy": acc,
                "f1_macro": f1
            })

        pd.DataFrame(results).sort_values("accuracy", ascending=False).head(10)