import os
import pandas as pd

from datm.data_tools.retrieval.django_env import DJANGO_PROJECT_PATH
from datm.data_tools.django.models import Dataset


class ProjectDataset(object):

    def __init__(self, project_id, dataset_id):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset = Dataset.objects.get(project_asset__project__id=project_id, project_asset__id=dataset_id)

    def df(self):
        return pd.read_csv(self.csv_path)

    @property
    def csv_path(self):
        csv_path = os.path.join(DJANGO_PROJECT_PATH, self.dataset.csv.name)
        return csv_path
