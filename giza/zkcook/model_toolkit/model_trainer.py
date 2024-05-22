class ModelTrainer:
    def __init__(self, model_class, model_type):
        self.model_class = model_class
        self.model_type = model_type
        self.model_trainers_by_name = {
            "xgboost": self.train_xgb,
            "lightgbm": self.train_lgb,
            "catboost": self.train_catboost,
        }

    def train_xgb(
        self,
        X_train,
        y_train,
        X_eval,
        y_eval,
        params,
        eval_metric="auc",
        early_stopping_rounds=10,
    ):
        params["eval_metric"] = eval_metric
        params["early_stopping_rounds"] = early_stopping_rounds
        model = self.model_class(**params)
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)

        return model

    def train_lgb(
        self,
        X_train,
        y_train,
        X_eval,
        y_eval,
        params,
        eval_metric="auc",
        early_stopping_rounds=10,
    ):
        fit_params = {key: val for key, val in params.items() if key not in ["silent", "eval_metric", "early_stopping_rounds"]}
        fit_params["verbose"] = -1
        fit_params["early_stopping_rounds"] = early_stopping_rounds
        model = self.model_class(**fit_params)
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric=eval_metric)

        return model

    def train_catboost(
        self,
        X_train,
        y_train,
        X_eval,
        y_eval,
        params,
        eval_metric="AUC",
        early_stopping_rounds=10,
    ):
        params["early_stopping_rounds"] = early_stopping_rounds
        params["verbose"] = 0
        params["eval_metric"] = eval_metric
        model = self.model_class(**params)

        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)])

        return model

    def train_model(
        self,
        X_train,
        y_train,
        X_eval,
        y_eval,
        params,
        eval_metric="auc",
        early_stopping_rounds=10,
    ):
        if self.model_type in self.model_trainers_by_name:
            train_func = self.model_trainers_by_name[self.model_type]
            trained_model = train_func(
                X_train,
                y_train,
                X_eval,
                y_eval,
                params,
                eval_metric=eval_metric,
                early_stopping_rounds=early_stopping_rounds,
            )
            return trained_model
        else:
            raise ValueError(f"No training function found for model type {self.model_type}.")
