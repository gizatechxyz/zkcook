XGBOOST_METRICS = {
    "rmse": "minimize",  # Root Mean Square Error
    "rmsle": "minimize",  # Root Mean Square Logarithmic Error
    "mae": "minimize",  # Mean Absolute Error
    "mape": "minimize",  # Mean Absolute Percentage Error
    "mphe": "minimize",  # Mean Pseudo Huber Error
    "logloss": "minimize",  # Negative Log-Likelihood
    "error": "minimize",  # Binary classification error rate
    "merror": "minimize",  # Multiclass classification error rate
    "mlogloss": "minimize",  # Multiclass logloss
    "auc": "maximize",  # Area Under the Curve
    "aucpr": "maximize",  # Area under the PR curve
    "ndcg": "maximize",  # Normalized Discounted Cumulative Gain
    "map": "maximize",  # Mean Average Precision
    "pre": "maximize",  # Precision at ranks list
    "poisson-nloglik": "minimize",  # Negative log-likelihood for Poisson regression
    "gamma-nloglik": "minimize",  # Negative log-likelihood for gamma regression
    "cox-nloglik": "minimize",  # Negative partial log-likelihood for Cox proportional hazards regression
    "gamma-deviance": "minimize",  # Residual deviance for gamma regression
    "tweedie-nloglik": "minimize",  # Negative log-likelihood for Tweedie regression
    "aft-nloglik": "minimize",  # Negative log likelihood for Accelerated Failure Time model
    "interval-regression-accuracy": "maximize",  # Accuracy for interval-censored data in AFT models
}

LIGHTGBM_METRICS = {
    "l1": "minimize",  # Mean Absolute Error (MAE)
    "mean_absolute_error": "minimize",
    "mae": "minimize",
    "regression_l1": "minimize",
    "l2": "minimize",  # Mean Squared Error (MSE)
    "mean_squared_error": "minimize",
    "mse": "minimize",
    "regression_l2": "minimize",
    "regression": "minimize",
    "rmse": "minimize",  # Root Mean Squared Error (RMSE)
    "root_mean_squared_error": "minimize",
    "l2_root": "minimize",
    "quantile": "minimize",  # Quantile regression
    "mape": "minimize",  # Mean Absolute Percentage Error
    "mean_absolute_percentage_error": "minimize",
    "huber": "minimize",  # Huber loss
    "fair": "minimize",  # Fair loss
    "poisson": "minimize",  # Poisson regression
    "gamma": "minimize",  # Gamma regression
    "gamma_deviance": "minimize",
    "tweedie": "minimize",  # Tweedie regression
    "ndcg": "maximize",  # Normalized Discounted Cumulative Gain
    "lambdarank": "maximize",
    "rank_xendcg": "maximize",
    "xendcg": "maximize",
    "xe_ndcg": "maximize",
    "xe_ndcg_mart": "maximize",
    "xendcg_mart": "maximize",
    "map": "maximize",  # Mean Average Precision
    "mean_average_precision": "maximize",
    "auc": "maximize",  # Area Under the Curve
    "average_precision": "maximize",
    "binary_logloss": "minimize",  # Binary classification log loss
    "binary": "minimize",
    "binary_error": "minimize",  # Binary classification error
    "auc_mu": "maximize",  # AUC-mu
    "multi_logloss": "minimize",  # Multiclass logloss
    "multiclass": "minimize",
    "softmax": "minimize",
    "multiclassova": "minimize",
    "multiclass_ova": "minimize",
    "ova": "minimize",
    "ovr": "minimize",
    "multi_error": "minimize",  # Multiclass error rate
    "cross_entropy": "minimize",  # Cross-entropy
    "xentropy": "minimize",
    "cross_entropy_lambda": "minimize",  # Intensity-weighted cross-entropy
    "xentlambda": "minimize",
    "kullback_leibler": "minimize",  # Kullback-Leibler divergence
    "kldiv": "minimize",
}


def check_metric_optimization(model_type, metric):
    if model_type.lower() == "xgboost":
        if metric in XGBOOST_METRICS:
            return XGBOOST_METRICS[metric]
        else:
            raise ValueError("The metric is not supported by XGBoost.")
    elif model_type.lower() == "lightgbm":
        if metric in LIGHTGBM_METRICS:
            return LIGHTGBM_METRICS[metric]
        else:
            raise ValueError("The metric is not supported by LightGBM.")
    else:
        raise ValueError("Invalid model type. Only 'xgboost' and 'lightgbm' are supported.")
