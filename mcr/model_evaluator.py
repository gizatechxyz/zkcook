import json

class ModelEvaluator:
    def __init__(self, model_type, eval_metric):
        self.eval_metric = eval_metric
        self.model_type = model_type
        self.model_evaluators_by_name = {
            'xgboost': self.evaluate_xgb,
            'lightgbm': self.evaluate_lgb,
            'catboost': self.evaluate_catboost,
        }

    def evaluate_xgb(self, model):
        best_score = model.best_score
        return best_score

    def evaluate_lgb(self, model):
        best_score = model.best_score_
        best_score = best_score['valid_0'][self.eval_metric]
        return best_score

    def evaluate_catboost(self, model):
        best_score = model.get_best_score()['validation'][self.eval_metric]
        return best_score

    def evaluate_model(self, model):
        if self.model_type in self.model_evaluators_by_name:
            evaluate_func = self.model_evaluators_by_name[self.model_type]
            best_score = evaluate_func(model)
            return best_score
        else:
            raise ValueError(f"No evaluation function found for model type {self.model_type}.")
