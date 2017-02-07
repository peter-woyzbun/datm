import os
import sys

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

USER_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_datasets")
USER_VISUALIZATIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_visualizations")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db.sqlite3")


def db_exists():
    """ Check if the datm SQLite database exists. """
    return os.path.exists(DB_PATH)
