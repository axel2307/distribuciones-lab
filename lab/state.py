"""Plain-Python state models; Streamlit widget state is only a view."""
from copy import deepcopy
from .catalog import defaults


def initial_event(distribution_id, params):
    if distribution_id == "multinomial":
        counts = [int(params["n"] * p) for p in params["probabilities"]]
        counts[-1] += params["n"] - sum(counts)
        return {"kind": "joint", "counts": counts}
    from .calculations import quantile
    return {"kind": "le", "x": quantile(distribution_id, params, 0.5)}


def new_model(distribution_id, parameters=None, event=None):
    params = deepcopy(parameters if parameters is not None else defaults(distribution_id))
    return {"parameters": params, "event": deepcopy(event if event is not None else initial_event(distribution_id, params)),
            "coverage": 0.99, "category": 0}


def get_model(models, key, distribution_id, parameters=None, event=None):
    if key not in models:
        models[key] = new_model(distribution_id, parameters, event)
    return models[key]


def replace_model(models, key, distribution_id, parameters=None, event=None):
    models[key] = new_model(distribution_id, parameters, event)
    return models[key]
