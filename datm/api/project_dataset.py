import os
import pandas as pd

from datm.web import PROJECT_PATH
from datm.api.django.models import Dataset, Project


class ProjectDataset(object):

    def __init__(self, project_id, dataset_id):
        self.project = Project.objects.get(id=project_id)
        self.dataset = Dataset.objects.get(project=self.project, project_asset__id=dataset_id)

    def df(self):
        return self.dataset.df

    @property
    def csv_path(self):
        csv_path = os.path.join(PROJECT_PATH, self.dataset.csv.name)
        return csv_path
