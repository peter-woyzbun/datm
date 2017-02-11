from pandasql import sqldf
import pandas as pd

from datm.data_tools.transformations.base import DataTransformation


class SqlQuery(DataTransformation):

    def __init__(self, query, joinable_dataset_map, source_code_mode=False):
        """
        Initialize the SqlQuery instance and immediately register
        any joins contained in the query string.

        Parameters
        ----------
        query : str
            The actual SQL query.
        joinable_dataset_map : dict
            A dictionary mapping 'joinable' dataset (those that wont cause
            a cycle in the project graph if joined) names to their IDs.
            Ex: '{'some_dataset_name': 69}'
        source_code_mode : bool
            Whether or not to return the source code required to execute
            the transformation, rather than performing the transformation.

        """
        self.query = query
        self.joinable_dataset_map = joinable_dataset_map

        super(SqlQuery, self).__init__(source_code_mode=source_code_mode)

        self._register_joins()

    @property
    def joinable_dataset_names(self):
        return self.joinable_dataset_map.keys()

    def _register_joins(self):
        """
        Search the query to find any reference to 'joinable' dataset names,
        which would indicate a join with that table.

        """
        for dataset_name in self.joinable_dataset_names:
            if dataset_name in self.query:
                self.register_join(self.joinable_dataset_map[dataset_name])

    @staticmethod
    def _unicode_col_fix(df):
        """
        Fixes the error "TypeError: [unicode] is not implemented as a table column" when
        writing to HDF - not sure why this is required (?) with pandasql.

        """
        types = df.apply(lambda x: pd.lib.infer_dtype(x.values))
        for col in types[types == 'unicode'].index:
            df[col] = df[col].astype(str)

        df.columns = [str(c) for c in df.columns]
        return df

    def _execute(self, tables):
        df = sqldf(self.query, tables)
        df = self._unicode_col_fix(df)
        return df

    @staticmethod
    def _tables_dict_source(tables):
        """ Create an string representation of table dictionary that can be evaluated. """
        dict_entries = list()
        for k, v in tables.items():
            dict_entries.append("'%s': %s" % (k, v))
        dict_body = ", ".join(dict_entries)
        dict_source = "{" + dict_body + "}"
        return dict_source

    def _source_code_execute(self, tables):
        tables_src = self._tables_dict_source(tables)
        source_str = "sqldf(\"\"\"%s\"\"\", %s)" % (self.query, tables_src)
        return source_str
