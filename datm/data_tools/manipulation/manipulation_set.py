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

    def __init__(self, evaluator, df_mutable=False):
        self.evaluator = evaluator
        self.df_mutable = df_mutable
        self.execution_successful = True

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def execute(self, df):
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

    @abstractmethod
    def _execute(self, df):
        raise NotImplementedError

    @staticmethod
    def add_name_to_evaluator(evaluator, name, name_val):
        evaluator.add_name(name, name_val)

    @staticmethod
    def remove_name_from_evaluator(evaluator, name):
        del evaluator.names[name]


class Select(Manipulation):

    def __init__(self, evaluator, columns):
        self.columns = columns
        super(Select, self).__init__(evaluator=evaluator, df_mutable=True)

    def _execute(self, df):
        cols = self.columns.replace(" ", "")
        column_lis = cols.split(",")
        df = df[column_lis]
        return df


class Filter(Manipulation):

    def __init__(self, evaluator, conditions):
        self.conditions = conditions
        super(Filter, self).__init__(evaluator=evaluator, df_mutable=True)

    def _execute(self, df):
        # Todo: non-numeric value handling for evaluator.
        conditions = self.conditions.split(",")
        for condition in conditions:
            df = df[self.evaluator.eval(condition)]
        return df


class Create(Manipulation):

    def __init__(self, evaluator, new_column_name, new_column_definition):
        self.new_column_name = new_column_name
        self.new_column_definition = new_column_definition
        super(Create, self).__init__(evaluator=evaluator, df_mutable=True)

    @timeit
    def _execute(self, df):
        if self.new_column_name == 'df':
            df = self.evaluator.eval(self.new_column_definition)
        else:
            df[self.new_column_name] = self.evaluator.eval(self.new_column_definition)
        return df


class Rename(Manipulation):

    def __init__(self, evaluator, old_column_name, new_column_name):
        self.old_column_name = old_column_name
        self.new_column_name = new_column_name
        super(Rename, self).__init__(evaluator=evaluator, df_mutable=True)

    def _execute(self, df):
        df = df.rename(columns={self.old_column_name: self.new_column_name})
        return df


class Slice(Manipulation):

    def __init__(self, evaluator, from_row, to_row):
        self.from_row = from_row
        self.to_row = to_row
        super(Slice, self).__init__(evaluator=evaluator, df_mutable=False)

    def _execute(self, df):
        self.add_name_to_evaluator(evaluator=self.evaluator,
                                   name='n',
                                   name_val=len(df.index))

        return df[self.evaluator.eval(self.from_row):self.evaluator.eval(self.to_row)]


class GroupBy(Manipulation):

    def __init__(self, evaluator, columns, aggregations):
        self.columns = columns
        self.aggregations = aggregations
        super(GroupBy, self).__init__(evaluator=evaluator, df_mutable=True)

    @property
    def formatted_aggregations(self):
        aggregation_dict = defaultdict(dict)
        for aggregation in self.aggregations:
            aggregation_dict[aggregation['target_column']][aggregation['column_name']] = aggregation['summary_type']
        return aggregation_dict

    def _execute(self, df):
        cols = self.columns.replace(" ", "")
        column_lis = cols.split(",")
        aggregations = self.formatted_aggregations
        df = df.groupby(column_lis, as_index=False).agg(aggregations)
        mi = df.columns
        new_index_lis = []
        # Todo: Find better way of doing this...
        for e in mi.tolist():
            if e[1] == '':
                new_index_lis.append(e[0])
            else:
                new_index_lis.append(e[1])

        ind = pd.Index(new_index_lis)
        df.columns = ind
        return df


class Join(Manipulation):

    def __init__(self, evaluator, target_dataset_id, join_type, on_columns):
        self.target_dataset_id = target_dataset_id
        self.target_df = None
        self.join_type = join_type
        self.on_columns = on_columns
        self.left_cols = list()
        self.right_cols = list()
        super(Join, self).__init__(evaluator=evaluator, df_mutable=True)

    def _add_column_name(self, col_name, col_side):
        if col_side == 'left':
            self.left_cols.append(col_name)
        elif col_side == 'right':
            self.right_cols.append(col_name)

    def _parse_on_columns_str(self):
        column_pair = (Word(alphas, alphanums + "_$").setParseAction(lambda x: self._add_column_name(col_name=x[0], col_side='left'))
                       + Suppress("=") + Word(alphas, alphanums + "_$").setParseAction(lambda x: self._add_column_name(col_name=x[0],
                                                                                                    col_side='right')))
        col_pair_list = delimitedList(column_pair)
        col_pair_list.parseString(self.on_columns)

    def _execute(self, df):
        if self.target_df is not None:
            self._parse_on_columns_str()
            return df.merge(self.target_df, how=self.join_type, left_on=self.left_cols, right_on=self.right_cols)
        else:
            return df


class SortBy(Manipulation):

    def __init__(self, evaluator, columns):
        self.columns = columns
        self.ascending = list()
        self.column_list = list()
        super(SortBy, self).__init__(evaluator=evaluator, df_mutable=False)

    def _add_col_to_list(self, col_name, ascending=True):
        self.column_list.append(col_name)
        if ascending:
            self.ascending.append(1)
        else:
            self.ascending.append(0)

    def _parse_columns(self):
        minus = Suppress("-")
        descending_col = (minus + Word(alphas, alphanums + "_$")).setParseAction(lambda x: self._add_col_to_list(col_name=x[0], ascending=False))
        ascending_col = Word(alphas, alphanums + "_$").setParseAction(lambda x: self._add_col_to_list(col_name=x[0]))
        column_list = delimitedList(descending_col | ascending_col)
        column_list.parseString(self.columns)

    def _execute(self, df):
        self._parse_columns()
        df = df.sort_values(self.column_list, ascending=self.ascending)
        return df


class WideToLong(Manipulation):

    def __init__(self, evaluator, value_columns, id_columns=None):
        self.id_columns = id_columns
        self.value_columns = value_columns
        super(WideToLong, self).__init__(evaluator=evaluator, df_mutable=True)

    def _execute(self, df):
        value_vars = self.value_columns.replace(" ", "").split(",")
        if self.id_columns is not None:
            id_vars = self.id_columns.replace(" ", "").split(",")
            df = pd.melt(df, id_vars=id_vars, value_vars=value_vars)
        else:
            df = pd.melt(df, value_vars=value_vars)
        return df


class ManipulationSet(object):

    MANIPULATION_TYPES = {'select': Select,
                          'filter': Filter,
                          'create': Create,
                          'rename': Rename,
                          'slice': Slice,
                          'group_by': GroupBy,
                          'join': Join,
                          'sort_by': SortBy,
                          'wide_to_long': WideToLong}

    EVAL_REQUIRED = ('create',
                     'filter',
                     'group_by',
                     'slice',
                     'rename',
                     'wide_to_long')

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.manipulations = list()
        self.error_data = dict(invalid_manipulations=list())
        self.join_manipulations = list()
        self.invalid_manipulations = list()
        self.column_dtype_set = None
        self.join_dfs = None

    def __add__(self, other):
        self.manipulations.append(other)
        return self

    __radd__ = __add__

    def __iter__(self):
        for x in self.manipulations:
            yield x

    def register_join_manipulation(self, join_manipulation):
        self.join_manipulations.append(join_manipulation)

    @classmethod
    def create_from_list(cls, manipulation_list):
        evaluator = ArgEvaluator()
        manipulation_set = ManipulationSet(evaluator=evaluator)
        for manipulation_dict in manipulation_list:
            manipulation_type = manipulation_dict.pop('manipulation_type', None)
            # Special handling for join manipulations.
            if manipulation_type == 'join':
                join_manipulation = Join(evaluator, **manipulation_dict)
                manipulation_set.register_join_manipulation(join_manipulation)
                manipulation_set += join_manipulation
            else:
                manipulation = cls.MANIPULATION_TYPES[manipulation_type](evaluator, **manipulation_dict)
                manipulation_set += manipulation
        return manipulation_set

    def collect_errors(self):
        manipulation_index = 0
        for manipulation in self:
            if not manipulation.execution_successful:
                self.error_data['invalid_manipulations'].append(manipulation_index)
            manipulation_index += 1

    @timeit
    def execute(self, df):
        self.evaluator.update_names(df)
        pre_transform_type_set = ColumnDTypeSet(df=df.copy())
        for manipulation in self:
            df = manipulation.execute(df)
        post_transformation_type_set = ColumnDTypeSet(df=df)
        self.column_dtype_set = post_transformation_type_set - pre_transform_type_set
        self.collect_errors()
        return df

