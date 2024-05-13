# auto-zkml

This package is designed to provide functionality that facilitates the transition from ML algorithms to ZKML. Its two main functionalities are:

- [**Serialization**](#serialization): saving a trained ML model in a specific format to be interpretable by other programs.

- [**model-complexity-reducer (mcr)**](#mcr): Given a model and a training dataset, transform the model and the data to obtain a lighter representation that maximizes the tradeoff between performance and complexity.

It's important to note that although the main goal is the transition from ML to ZKML, auto-zkml can be useful in other contexts, such as:

- The model's weight needs to be minimal, for example for mobile applications.
- Minimal inference times are required for low latency applications.
- We want to check if we have created an overly complex model and a simpler one would give us the same performance (or even better).
- The number of steps required to perform the inference must be less than X (as is currently constrained by the ZKML paradigm).

## Installation

### Install from PyPi

For the latest release:

```bash
pip install auto-zkml
```

### Installing from source

Clone the repository and install it with `pip`:


```bash
    git clone git@github.com:gizatechxyz/auto-zkml.git
    cd auto-zkml
    pip install .
```

## Serialization

To see in more detail how this tool works, check out this [tutorial](tutorials/serialize_my_model.ipynb).

To run it:

```python
from auto_zkml import serialize_model

serialize_model(YOUR_TRAINED_MODEL, "OUTPUT_PATH/MODEL_NAME.json")
```

## mcr

To see in more detail how this tool works, check out this [tutorial](tutorials/reduce_model_complexity.ipynb).

To run it:

```python
model, transformer = mcr(model = MY_MODEL,
                         X_train = X_train,
                         y_train = y_train,
                         X_eval = X_test,
                         y_eval = y_test,
                         eval_metric = 'rmse',
                         transform_features = True)
```


### Supported Models

|        Model        | status |
| :-----------------: | :---:  |
|    XGBRegressor     |   ✅   |
|    XGBClassifier    |   ✅   |
|    LGBMRegressor    |   ✅   |
|    LGBMClassifier   |   ✅   |
| Logistic Regression |   ⏳   |
|        GARCH        |   ⏳   |
