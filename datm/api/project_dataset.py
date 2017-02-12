import os
import pandas as pd

from datm.web import PROJECT_PATH
from datm.api.django.models import Dataset, Project


class ProjectDataset(object):

    def __init__(self, project_id, dataset_id):
        self._project = Project.objects.get(id=project_id)
        self._dataset = Dataset.objects.get(project=self._project, project_asset__id=dataset_id)

    def df(self):
        return self._dataset.df