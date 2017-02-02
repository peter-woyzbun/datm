import pandas as pd
import numpy as np
import operator

from datm.data_tools.manipulation import eval_funcs


class ArgEvaluator(object):

    EVAL_FUNCTIONS = {
        # Numpy functions.
        "exp": np.exp,
        "expm1": np.expm1,
        "log": np.log,
        "log10": np.log10,
        "cos": np.cos,
        "hypot": np.hypot,
        "sin": np.sin,
        "tan": np.tan,
        "mean": np.mean,
        "std": np.std,
        "round": np.round,
        "floor": np.floor,
        "ceiling": np.ceil,
        "cumsum": np.cumsum,
        # Standard Python functions.
        "sum": sum,
        "pow": pow,
        # Custom functions.
        "rollmean": eval_funcs.rollmean,
        "lag": eval_funcs.lag,
        "as_string": eval_funcs.as_string,
        "as_int": eval_funcs.as_int,
        "as_date": eval_funcs.as_date,
        "add_lead_zeroes": eval_funcs.add_lead_zeroes,
        "day_of_week": eval_funcs.day_of_week,
        "category_bins": eval_funcs.category_bins,
        "replace": eval_funcs.replace,
        "drop_na": eval_funcs.drop_na
    }

    def __init__(self):
        self.names = dict()

    @property
    def _locals(self):
        locals_dict = dict(self.EVAL_FUNCTIONS.items() + self.names.items())
        return locals_dict

    def update_names(self, df):
        col_names = df.columns.values.tolist()
        names_dict = {c: df[c] for c in col_names}
        names_dict['df'] = df
        self.names = names_dict

    def add_name(self, name, name_val):
        self.names[name] = name_val

    def eval(self, eval_string):
        return eval(eval_string, self._locals)