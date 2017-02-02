from pandasql import sqldf


class DfSqlQuery(object):

    def __init__(self, query, joinable_datasets):
        self.query = query
        self.joinable_datasets = joinable_datasets

    def required_datasets(self):
        req_list = list()
        for dataset_name in self.joinable_datasets:
            if dataset_name in self.query:
                req_list.append(dataset_name)
        return req_list

    def execute(self, tables):
        return sqldf(self.query, tables)