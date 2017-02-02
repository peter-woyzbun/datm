from pandasql import sqldf

from datm.data_tools.transformations.base import DataTransformation


class SqlQuery(DataTransformation):

    def __init__(self, query, joinable_dataset_map, source_code_mode=False):
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
        source_str = "sqldf(\"\"\"%s\"\"\", %s)" % (self.query, tables)
        return source_str
