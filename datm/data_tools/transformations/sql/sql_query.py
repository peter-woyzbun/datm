from pandasql import sqldf

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

    def _execute(self, tables):
        return sqldf(self.query, tables)

    def _source_code_execute(self, tables):
        tables_str = str(tables)
        for df_name in tables.values():
            tables_str.replace(old="'%s'" % df_name, new=df_name)
        source_str = "sqldf(\"\"\"%s\"\"\", %s)" % (self.query, tables_str)
        return source_str
