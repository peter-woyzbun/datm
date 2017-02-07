import os
import sys
import importlib
import django

from datm.web import PROJECT_PATH


sys.path.append(PROJECT_PATH)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings')
django.setup()

models = importlib.import_module("core.models")

Dataset = models.Dataset
Project = models.Project
