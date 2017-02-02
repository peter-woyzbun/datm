from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pyparsing import Word, Literal, delimitedList, alphas, alphanums, Suppress
import pandas as pd

from datm.data_tools.manipulation.evaluator import ArgEvaluator
from datm.data_tools.manipulation.df_col_typing import ColumnDTypeSet
from datm.utils.func_timer import timeit


DEBUG_ENABLED = True


class Manipulation(object):
    """
    Base class for all manipulation types.


    """

    error_message = None

    def __init__(self, manipulation_set, df_mutable=False):
        self.manipulation_set = manipulation_set
        self.evaluator = manipulation_set.evaluator
        self.df_mutable = df_mutable
        self.execution_successful = True

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def execute(self, df):
        if not self.manipulation_set.source_code_mode:
            if not DEBUG_ENABLED:
                try:
                    df = self._execute(df=df)
                    if self.df_mutable:
                        self.evaluator.update_names(df)
                    return df
                except:
                    self.execution_successful = False
                    return df
            else:
                df = self._execute(df=df)
                if self.df_mutable:
                    self.evaluator.update_names(df)
                return df
        else:
            source_str = self._source_code_execute(df)
            return "df = %s" % source_str

    @abstractmethod
    def _execute(self, df):
        raise NotImplementedError

    @abstractmethod
    def _source_code_execute(self, df):
        raise NotImplementedError

    @staticmethod
    def add_name_to_evaluator(evaluator, name, name_val):
        evaluator.add_name(name, name_val)

    @staticmethod
    def remove_name_from_evaluator(evaluator, name):
        del evaluator.names[name]


class Select(Manipulation):

    def __init__(self, manipulation_set, columns):
        self.columns = columns
        super(Select, self).__init__(manipulation_set=manipulation_set, df_mutable=True)

    def _execute(self, df):
        cols = self.columns.replace(" ", "")
        column_lis = cols.split(",")
        df = df[column_lis]
        return df

    def _source_code_execute(self, df):
        src_string = "df = df[%s]" % str(self.columns)
        return src_string


class Filter(Manipulation):

    def __init__(self, manipulation_set, conditions):
        self.conditions = conditions
        super(Filter, self).__init__(manipulation_set=manipulation_set, df_mutable=True)

    def _execute(self, df):
        # Todo: non-numeric value handling for evaluator.
        conditions = self.conditions.split(",")
        for condition in conditions:
            df = df[self.evaluator.eval(condition)]
        return df

    def _source_code_execute(self, df):

        conditions = self.conditions.split(",")
        for condition in conditions:
            df = df[self.evaluator.eval(condition)]
            source_str = "df = df[%s]" % (self.evaluator.eval())