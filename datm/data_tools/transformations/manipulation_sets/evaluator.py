import numpy as np
import pandas as pd

# =============================================
# Eval Object
# ---------------------------------------------


# =============================================
# Eval Functions
# ---------------------------------------------

class EvalFunction(object):

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.source_str = ""
        self.dataset_label = "%s_df" % evaluator.dataset_name

    def __call__(self, *args, **kwargs):
        if self.evaluator.source_code_mode:
            new_instance = self.__class__(evaluator=self.evaluator)
            new_instance.source_str += str(self._source_code_execute(*args, **kwargs))
            return new_instance
        else:
            return self._execute(*args, **kwargs)

    def _execute(self, *args, **kwargs):
        """
        Method that should return the actual result of
        executing the evaluation function.

        """
        raise NotImplementedError

    def _source_code_execute(self, *args, **kwargs):
        """
        Method that should return the source code for the
        evaluation function.

        """
        raise NotImplementedError

    def __str__(self):
        return self.source_str

    @staticmethod
    def _join_src_strings(left_str, op, right_str):
        return "%s %s %s" % (left_str, op, right_str)

    def _op_join_instances(self, left_instance, op, right_instance):
        new_instance = self.__class__(evaluator=self.evaluator)
        if type(right_instance) is str:
            new_instance.source_str += "%s %s '%s'" % (str(left_instance), op, right_instance)
        else:
            new_instance.source_str += "%s %s %s" % (str(left_instance), op, str(right_instance))
        return new_instance

    # ----------------------
    # OPERATORS
    # ----------------------

    def __eq__(self, other):
        return self._op_join_instances(left_instance=self, op="==", right_instance=other)

    def __ne__(self, other):
        return self._op_join_instances(left_instance=self, op="!=", right_instance=other)

    def __add__(self, other):
        return self._op_join_instances(left_instance=self, op="+", right_instance=other)

    def __sub__(self, other):
        return self._op_join_instances(left_instance=self, op="-", right_instance=other)

    def __lt__(self, other):
        return self._op_join_instances(left_instance=self, op="<", right_instance=other)

    def __le__(self, other):
        return self._op_join_instances(left_instance=self, op="<=", right_instance=other)

    def __gt__(self, other):
        return self._op_join_instances(left_instance=self, op=">", right_instance=other)

    def __ge__(self, other):
        return self._op_join_instances(left_instance=self, op=">=", right_instance=other)

    def __mul__(self, other):
        return self._op_join_instances(left_instance=self, op="*", right_instance=other)

    def __div__(self, other):
        return self._op_join_instances(left_instance=self, op="/", right_instance=other)

    def __pow__(self, power, modulo=None):
        new_instance = self.__class__(evaluator=self.evaluator)
        new_instance.source_str += "(%s)**%s" % (str(self), power)
        return new_instance


class Log(EvalFunction):

    def __init__(self, evaluator):
        super(Log, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.log(col)

    def _source_code_execute(self, col):
        return "np.log(%s)" % col


class Log10(EvalFunction):

    def __init__(self, evaluator):
        super(Log10, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.log10(col)

    def _source_code_execute(self, col):
        return "np.log10(%s)" % col


class Exp(EvalFunction):

    def __init__(self, evaluator):
        super(Exp, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.exp(col)

    def _source_code_execute(self, col):
        return "np.exp(%s)" % col


class ExpM1(EvalFunction):

    def __init__(self, evaluator):
        super(ExpM1, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.expm1(col)

    def _source_code_execute(self, col):
        return "np.expm1(%s)" % col


class Cos(EvalFunction):

    def __init__(self, evaluator):
        super(Cos, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.cos(col)

    def _source_code_execute(self, col):
        return "np.cos(%s)" % col


class Mean(EvalFunction):

    def __init__(self, evaluator):
        super(Mean, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.mean(col)

    def _source_code_execute(self, col):
        return "np.mean(%s)" % col


class Std(EvalFunction):

    def __init__(self, evaluator):
        super(Std, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.std(col)

    def _source_code_execute(self, col):
        return "np.std(%s)" % col


class Round(EvalFunction):

    def __init__(self, evaluator):
        super(Round, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.round(col)

    def _source_code_execute(self, col):
        return "np.round(%s)" % col


class Max(EvalFunction):

    def __init__(self, evaluator):
        super(Max, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return max(col)

    def _source_code_execute(self, col):
        return "max(%s)" % col


class Min(EvalFunction):

    def __init__(self, evaluator):
        super(Min, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return min(col)

    def _source_code_execute(self, col):
        return "min(%s)" % col


class Floor(EvalFunction):

    def __init__(self, evaluator):
        super(Floor, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.floor(col)

    def _source_code_execute(self, col):
        return "np.floor(%s)" % col


class Ceiling(EvalFunction):

    def __init__(self, evaluator):
        super(Ceiling, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.ceil(col)

    def _source_code_execute(self, col):
        return "np.ceil(%s)" % col


class CumSum(EvalFunction):

    def __init__(self, evaluator):
        super(CumSum, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return np.cumsum(col)

    def _source_code_execute(self, col):
        return "np.cumsum(%s)" % col


class Sum(EvalFunction):

    def __init__(self, evaluator):
        super(Sum, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return sum(col)

    def _source_code_execute(self, col):
        return "sum(%s)" % col


class RollMean(EvalFunction):

    def __init__(self, evaluator):
        super(RollMean, self).__init__(evaluator=evaluator)

    def _execute(self, col, n):
        return pd.rolling_mean(col, n)

    def _source_code_execute(self, col, n):
        return "pd.rolling_mean(%s, %s)" % (col, n)


class Lag(EvalFunction):

    def __init__(self, evaluator):
        super(Lag, self).__init__(evaluator=evaluator)

    def _execute(self, col, n):
        return col.shift(n)

    def _source_code_execute(self, col, n):
        return "%s.shift(%s)" % (col, n)


class AsString(EvalFunction):

    def __init__(self, evaluator):
        super(AsString, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return col.astype(str)

    def _source_code_execute(self, col):
        return "%s.astype(str)" % col


class AsInt(EvalFunction):

    def __init__(self, evaluator):
        super(AsInt, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return col.astype(int)

    def _source_code_execute(self, col):
        return "%s.astype(int)" % col


class AsDate(EvalFunction):

    def __init__(self, evaluator):
        super(AsDate, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return pd.to_datetime(col)

    def _source_code_execute(self, col):
        return "pd.to_datetime(%s)" % col


class GetWeekday(EvalFunction):

    def __init__(self, evaluator):
        super(GetWeekday, self).__init__(evaluator=evaluator)

    def _execute(self, col):
        return col.dt.weekday_name

    def _source_code_execute(self, col):
        return "%s.dt.weekday_name" % col


class Replace(EvalFunction):

    def __init__(self, evaluator):
        super(Replace, self).__init__(evaluator=evaluator)

    def _execute(self, col, replacement_targets, replacement_values):
        return col.replace(replacement_targets, replacement_values, inplace=True)

    def _source_code_execute(self, col, replacement_targets, replacement_values):
        return "%s.replace(%s, %s, inplace=True)" % (col, str(replacement_targets), str(replacement_values))


class DropNa(EvalFunction):

    def __init__(self, evaluator):
        super(DropNa, self).__init__(evaluator=evaluator)

    def _execute(self, df):
        return df.dropna()

    def _source_code_execute(self, df):
        return "df.dropna()"


# =============================================
# Evaluator Class
# ---------------------------------------------

class Evaluator(object):

    def __init__(self, dataset_name, source_code_mode=False):
        self.dataset_name = dataset_name
        self.source_code_mode = source_code_mode
        self.funcs = {'log': Log(evaluator=self),
                      'log10': Log10(evaluator=self),
                      'exp': Exp(evaluator=self),
                      'expm1': ExpM1(evaluator=self),
                      'mean': Mean(evaluator=self),
                      'std': Std(evaluator=self),
                      'max': Max(evaluator=self),
                      'min': Min(evaluator=self),
                      'round': Round(evaluator=self),
                      'floor': Floor(evaluator=self),
                      'ceiling': Ceiling(evaluator=self),
                      'rollmean': RollMean(evaluator=self),
                      'lag': Lag(evaluator=self),
                      'as_string': AsString(evaluator=self),
                      'as_int': AsInt(evaluator=self),
                      'as_date': AsDate(evaluator=self),
                      'get_wk_day': GetWeekday(evaluator=self),
                      'replace': Replace(evaluator=self),
                      'drop_na': DropNa(evaluator=self)}
        self.names = dict()

    def update_names(self, df):
        col_names = df.columns.values.tolist()
        if self.source_code_mode:
            names_dict = {c: "%s_df['%s']" % (self.dataset_name, c) for c in col_names}
            names_dict['%s_df' % self.dataset_name] = 'df'
        else:
            names_dict = {c: df[c] for c in col_names}
            names_dict['df'] = df
        self.names = names_dict

    def add_source_col_name(self, col_name):
        """ Add a column name for source generation. """
        self.names[col_name] = "%s_df['%s']" % (self.dataset_name, col_name)

    @property
    def _locals(self):
        locals_dict = dict(self.funcs.items() + self.names.items())
        return locals_dict

    def eval(self, eval_string):
        if self.source_code_mode:
            return str(eval(eval_string, self._locals))
        else:
            return eval(eval_string, self._locals)

