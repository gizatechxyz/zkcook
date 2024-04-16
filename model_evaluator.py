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
        # XGBoost proporciona directamente la mejor iteración y la mejor puntuación
        best_iteration = model.best_iteration
        best_score = model.best_score

        return best_score, best_iteration

    def evaluate_lgb(self, model):
        # LightGBM: obtener la mejor iteración y luego buscar la mejor puntuación usando esta iteración
        best_iteration = model.best_iteration_
        best_score = model.best_score_
        best_score = best_score['valid_0'][self.eval_metric]
        return best_score, best_iteration

    def evaluate_catboost(self, model):
        # CatBoost: la mejor iteración se obtiene directamente, igual que la mejor puntuación
        best_iteration = model.get_best_iteration()
        # CatBoost almacena las métricas en minúsculas
        best_score = model.get_best_score()['validation'][self.eval_metric]
        return best_score, best_iteration

    def evaluate_model(self, model):
        if self.model_type in self.model_evaluators_by_name:
            evaluate_func = self.model_evaluators_by_name[self.model_type]
            best_score, best_iteration = evaluate_func(model)
            return best_score, best_iteration
        else:
            raise ValueError(f"No evaluation function found for model type {self.model_type}.")
