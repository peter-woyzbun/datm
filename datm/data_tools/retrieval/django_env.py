import os


DJANGO_PROJECT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'web')


def make_django_env():
    """
    Set up django environment so that django's ORM can be used outside
    of the project environment.

    """

    import sys
    django_project_path = DJANGO_PROJECT_PATH
    sys.path.append(django_project_path)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings')
    import django
    django.setup()


def django_dataset_model():
    make_django_env()
    import importlib
    models = importlib.import_module("core.models")
    return models.Dataset


def dataset_path():
    """
    Return the path to the directory that stores Dataset CSV files.

    """
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web')


