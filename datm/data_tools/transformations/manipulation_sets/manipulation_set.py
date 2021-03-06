import re
import pandas as pd
from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
from pyparsing import Word, Literal, delimitedList, alphas, alphanums, Suppress

import datm.config
from datm.data_tools.transformations.base import DataTransformation
from datm.data_tools.transformations.manipulation_sets.evaluator import Evaluator


class Manipulation(object):
    """
    Base class for all manipulation types.

    """

    error_message = None

    def __init__(self, manipulation_set, df_mutable=False):
        """
        Initializes the manipulation.

        Parameters
        ----------
        manipulation_set : ManipulationSet
            The ManipulationSet instance the manipulation is a member of.
        df_mutable : bool
            Whether or not the manipulation has an impact on the DataFrame's
            basic dimensions: column names, number of rows, etc...

        """
        if not isinstance(manipulation_set, ManipulationSet):
            raise TypeError
        else:
            self.manipulation_set = manipulation_set
        self.dataset_label = "%s_df" % self.manipulation_set.dataset_name
        self.evaluator = manipulation_set.evaluator
        self.df_mutable = df_mutable
        self.execution_successful = True

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def execute(self, df):
        if not self.manipulation_set.source_code_mode:
            # If debug is disabled, exceptions are 'hidden' so that
            # nothing breaks on the user's end - we simply pass the
            # DataFrame on to the next manipulation.
            if not datm.config.DEBUG_MANIPULATION_SETS:
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
            return source_str

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

    def __init__(self, manipulation_set, columns, remove_selected):
        self.columns = columns
        if remove_selected == "True":
            self.remove_selected = True
        elif remove_selected == "False":
            self.remove_selected = False
        super(Select, self).__init__(manipulation_set=manipulation_set, df_mutable=True)

    def _execute(self, df):
        cols = self.columns.replace(" ", "")
        column_lis = cols.split(",")
        if not self.remove_selected:
            df = df[column_lis]
        else:
            df = df.drop(column_lis, axis=1)
        return df

    def _source_code_execute(self, df):
        cols = self.columns.replace(" ", "")
        column_lis = cols.split(",")
        if not self.remove_selected:
            src_string = "%s = %s[[%s]]" % (self.dataset_label,
                                            self.dataset_label,
                                            ', '.join('"{0}"'.format(w) for w in column_lis))
        else:
            src_string = "%s = %s.drop(%s, axis=1)" % (self.dataset_label,
                                                       self.dataset_label,
                                                       ', '.join('"{0}"'.format(w) for w in column_lis))
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

    def _parsed_source_condition(self, condition):
        col_names_dict = self.manipulation_set.evaluator.names.copy()
        ordered_col_names = col_names_dict.keys()
        ordered_col_names.sort(key=len, reverse=True)
        pattern = re.compile('|'.join(ordered_col_names))
        result = pattern.sub(lambda x: col_names_dict[x.group()], condition)
        return result

    def _source_code_execute(self, df):
        source_strings = list()
        conditions = self.conditions.split(",")
        for condition in conditions:
            source_str = "%s = %s[%s]" % (self.dataset_label,
                                          self.dataset_label,
                                          self.evaluator.eval(condition))
            source_strings.append(source_str)
        source = "\n".join(source_strings)
        return source


class Mutate(Manipulation):

    def __init__(self, manipulation_set, new_column_name, new_column_definition):
        self.new_column_name = new_column_name
        self.new_column_definition = new_column_definition
        super(Mutate, self).__init__(manipulation_set=manipulation_set, df_mutable=True)

    def _execute(self, df):
        if self.new_column_name == 'df':
            print "Mutating dataframe!"
            df = self.evaluator.eval(self.new_column_definition)
            print df
        else:
            df[self.new_column_name] = self.evaluator.eval(self.new_column_definition)
        return df

    def _source_code_execute(self, df):
        # Todo: dataframe mutation handling.
        if self.new_column_name == 'df':
            source_str = "df = %s" % self.evaluator.eval(self.new_column_definition)
        else:
            source_str = "%s['%s'] = %s" % (self.dataset_label,
                                            self.new_column_name,
                                            self.evaluator.eval(self.new_column_definition))
            self.manipulation_set.evaluator.add_source_col_name(self.new_column_name)
        return source_str


class Rename(Manipulation):

    def __init__(self, manipulation_set, old_column_name, new_column_name):
        self.old_column_name = old_column_name
        self.new_column_name = new_column_name
        super(Rename, self).__init__(manipulation_set=manipulation_set, df_mutable=True)

    def _execute(self, df):
        df = df.rename(columns={self.old_column_name: self.new_column_name})
        return df

    def _source_code_execute(self, df):
        source_str = "%s = %s.rename(columns={'%s': '%s'})" % (self.dataset_label,
                                                               self.dataset_label,
                                                               self.old_column_name,
                                                               self.new_column_name)
        return source_str


class Join(Manipulation):

    def __init__(self, manipulation_set, target_dataset_id, join_type, on_columns):
        self.target_dataset_id = target_dataset_id
        self.target_df = None
        self.join_type = join_type
        self.on_columns = on_columns
        self.left_cols = list()
        self.right_cols = list()
        super(Join, self).__init__(manipulation_set=manipulation_set, df_mutable=True)
        # Register the join...
        self.manipulation_set.register_join(target_dataset_id)

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

    def _set_target_df(self):
        self.target_df = self.manipulation_set.join_dfs[int(self.target_dataset_id)]

    def _execute(self, df):
        self._set_target_df()
        if self.target_df is not None:
            self._parse_on_columns_str()
            return df.merge(self.target_df, how=self.join_type, left_on=self.left_cols, right_on=self.right_cols)
        else:
            return df

    def _source_code_execute(self, df):
        self._set_target_df()
        if self.target_df is not None:
            self._parse_on_columns_str()
        source_str = "%s = %s.merge(%s_df, how=%s, left_on=%s, right_on=%s)" % (self.dataset_label,
                                                                                self.dataset_label,
                                                                                self.target_df,
                                                                                "'" + self.join_type + "'",
                                                                                self.left_cols,
                                                                                self.right_cols)
        return source_str


class Slice(Manipulation):

    def __init__(self, manipulation_set, from_row, to_row):
        self.from_row = from_row
        self.to_row = to_row
        super(Slice, self).__init__(manipulation_set=manipulation_set, df_mutable=True)

    def _execute(self, df):
        self.add_name_to_evaluator(evaluator=self.evaluator,
                                   name='n',
                                   name_val=len(df.index))

        return df[self.evaluator.eval(self.from_row):self.evaluator.eval(self.to_row)]

    def _source_code_execute(self, df):
        self.add_name_to_evaluator(evaluator=self.evaluator,
                                   name='n',
                                   name_val=len(df.index))
        source_str = "%s = %s[%s:%s]" % (self.dataset_label,
                                         self.dataset_label,
                                         self.evaluator.eval(self.from_row),
                                         self.evaluator.eval(self.to_row))
        return source_str


class GroupBy(Manipulation):

    def __init__(self, manipulation_set, columns, aggregations):
        self.columns = columns
        self.aggregations = aggregations
        super(GroupBy, self).__init__(manipulation_set=manipulation_set, df_mutable=True)

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

    def _source_code_execute(self, df):
        aggregations = self.formatted_aggregations
        source_str = "%s = %s.groupby(%s, as_index=False).agg(%s)" % (self.dataset_label,
                                                                      self.dataset_label,
                                                                      self.columns,
                                                                      aggregations)
        return source_str


class SortBy(Manipulation):

    def __init__(self, manipulation_set, columns):
        self.columns = columns
        self.ascending = list()
        self.column_list = list()
        super(SortBy, self).__init__(manipulation_set=manipulation_set, df_mutable=False)

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

    def _source_code_execute(self, df):
        self._parse_columns()
        source_str = "%s = %s.sort_values(%s, ascending=%s)" % (self.dataset_label,
                                                                self.dataset_label,
                                                                self.column_list,
                                                                self.ascending)
        return source_str


class WideToLong(Manipulation):

    def __init__(self, manipulation_set, value_columns, id_columns=None):
        self.id_columns = id_columns
        self.value_columns = value_columns
        super(WideToLong, self).__init__(manipulation_set=manipulation_set, df_mutable=True)

    def _execute(self, df):
        value_vars = self.value_columns.replace(" ", "").split(",")
        if self.id_columns is not None:
            id_vars = self.id_columns.replace(" ", "").split(",")
            df = pd.melt(df, id_vars=id_vars, value_vars=value_vars)
        else:
            df = pd.melt(df, value_vars=value_vars)
        return df

    def _source_code_execute(self, df):
        value_vars = self.value_columns.replace(" ", "").split(",")
        if self.id_columns is not None:
            id_vars = self.id_columns.replace(" ", "").split(",")
            source_str = "%s = pd.melt(%s, id_vars=%s, value_vars=%s)" % (self.dataset_label,
                                                                          self.dataset_label,
                                                                          id_vars,
                                                                          value_vars)
        else:
            source_str = "%s = pd.melt(%s, value_vars=%s)" % (self.dataset_label, self.dataset_label, value_vars)
        return source_str


class LongToWide(Manipulation):

    def __init__(self, manipulation_set, columns_col, value_col, index_col=None):
        self.index_col = index_col
        self.columns_col = columns_col
        self.value_col = value_col

        super(LongToWide, self).__init__(manipulation_set=manipulation_set, df_mutable=True)

    def _execute(self, df):
        df = df.pivot(index=self.index_col, columns=self.columns_col, values=self.value_col)
        print df
        return df

    def _source_code_execute(self, df):
        source_str = "%s = %s.pivot(index=%s, columns=%s, values=%s)" % (self.dataset_label,
                                                                         self.dataset_label,
                                                                         self.index_col,
                                                                         self.columns_col,
                                                                         self.value_col)
        return source_str


class ManipulationSet(DataTransformation):

    MANIPULATION_TYPES = {'select': Select,
                          'filter': Filter,
                          'mutate': Mutate,
                          'rename': Rename,
                          'slice': Slice,
                          'group_by': GroupBy,
                          'join': Join,
                          'sort_by': SortBy,
                          'wide_to_long': WideToLong,
                          'long_to_wide': LongToWide}

    def __init__(self, dataset_name, evaluator, source_code_mode=False):
        self.dataset_name = dataset_name
        self.evaluator = evaluator
        self.manipulations = list()
        self.join_dfs = None
        super(ManipulationSet, self).__init__(source_code_mode=source_code_mode)
        self.error_data['invalid_manipulations'] = list()

    def __add__(self, other):
        self.manipulations.append(other)
        return self

    __radd__ = __add__

    def __iter__(self):
        for x in self.manipulations:
            yield x

    @classmethod
    def create_from_list(cls, dataset_name, manipulation_list, source_code_mode=False):
        """
        Create manipulation set from list of manipulation dicts. This is
        user provided data.

        """
        evaluator = Evaluator(dataset_name=dataset_name, source_code_mode=source_code_mode)
        manipulation_set = ManipulationSet(dataset_name=dataset_name,
                                           evaluator=evaluator,
                                           source_code_mode=source_code_mode)
        for manipulation_dict in manipulation_list:
            manipulation_type = manipulation_dict.pop('manipulation_type', None)
            manipulation = cls.MANIPULATION_TYPES[manipulation_type](manipulation_set, **manipulation_dict)
            manipulation_set += manipulation
        return manipulation_set

    def _execute(self, df):
        self.evaluator.update_names(df)
        for manipulation in self:
            df = manipulation.execute(df)
        self._collect_errors()
        return df

    def _source_code_execute(self, df):
        self.evaluator.update_names(df)
        source_strings = list()
        for manipulation in self:
            source_strings.append(manipulation.execute(df))
        source = "\n".join(source_strings)
        return source

    def _collect_errors(self):
        """ Check each manipulation in set for errors. """
        manipulation_index = 0
        for manipulation in self:
            if not manipulation.execution_successful:
                self.error_data['invalid_manipulations'].append(manipulation_index)
                self.execution_successful = False
            manipulation_index += 1

    def set_required_join_dfs(self, df_dict):
        self.join_dfs = df_dict
