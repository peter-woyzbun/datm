from abc import abstractmethod


class DataTransformation(object):

    def __init__(self, source_code_mode=False):
        self.source_code_mode = source_code_mode
        self.joined_dataset_ids = list()
        self.error_data = dict()
        self.execution_successful = True

    def register_join(self, dataset_id):
        self.joined_dataset_ids.append(dataset_id)

    def execute(self, *args, **kwargs):
        if not self.source_code_mode:
            return self._execute(*args, **kwargs)
        else:
            return self._source_code_execute(*args, **kwargs)

    @abstractmethod
    def _execute(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _source_code_execute(self, *args, **kwargs):
        raise NotImplementedError



