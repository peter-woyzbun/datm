# from __future__ import absolute_import
import os
import pandas as pd

from datm.data_tools.retrieval.django_env import django_dataset_model, DJANGO_PROJECT_PATH
from datm.utils.django_env import make_django_env

# make_django_env()

# from datm.web import setup
# setup()
# from datm.web.core.models import Dataset

# fun_dataset = Dataset.objects.get(id=55)
# print fun_dataset.name

# Dataset = django_dataset_model()

from datm.data_tools.django.models import Dataset


test_dataset = Dataset.objects.get(project_asset__name="Flights Cleaned")
print test_dataset

# csv_path = os.path.join(DJANGO_PROJECT_PATH, test_dataset.csv.name)

# print pd.read_csv(csv_path)
