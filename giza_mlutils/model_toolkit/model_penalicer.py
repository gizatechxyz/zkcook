class ModelPenalicer:
    def __init__(self, model_type):
        self.model_type = model_type
        self.penalicers_by_name = {
            "xgboost": self.penalicize_xgb,
            "lightgbm": self.penalicize_lgb,
            "catboost": self.penalicize_catboost,
        }

    def penalicize_xgb(self, model, final_eval, complexity_factor):
        trees = model.get_booster().get_dump()
        total_nodes = sum(tree.count("\n") for tree in trees)
        max_penalty = final_eval - (final_eval * complexity_factor)
        penalty = min(final_eval - (total_nodes * complexity_factor), max_penalty)

        return penalty

    def penalicize_lgb(self, model, final_eval, complexity_factor):
        trees = model.booster_.dump_model()["tree_info"]
        total_nodes = sum(tree["num_leaves"] + tree["num_cat"] for tree in trees)
        max_penalty = final_eval - (final_eval * complexity_factor)
        penalty = min(final_eval - (total_nodes * complexity_factor), max_penalty)
        return penalty

    def penalicize_catboost(self, model, final_eval, complexity_factor):
        trees = model.get_all_params()["tree_info"]
        total_nodes = sum(tree["num_leaves"] for tree in trees)
        max_penalty = final_eval - (final_eval * complexity_factor)
        penalty = min(final_eval - (total_nodes * complexity_factor), max_penalty)
        return penalty

    def penalize_model(self, model, final_eval, complexity_factor):
        if self.model_type in self.penalicers_by_name:
            penalicize_func = self.penalicers_by_name[self.model_type]
            penalty = penalicize_func(model, final_eval, complexity_factor)
            return penalty
        else:
            raise ValueError(f"No penalization function found for model type {self.model_type}.")
