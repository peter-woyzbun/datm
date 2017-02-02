import os
import sys
import importlib
import django

# Path to datm's django project directory.
DJANGO_PROJECT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'web')

# Setup the django environment.
django_project_path = DJANGO_PROJECT_PATH
sys.path.append(django_project_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings')
django.setup()

models = importlib.import_module("core.models")

Dataset = models.Dataset
Project = models.Project
