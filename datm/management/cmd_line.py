import click
import os
import StringIO
import webbrowser
import threading

from django.core import management

from datm.utils.django_env import in_django_env, make_django_env
from datm.data_tools.django.models import Dataset, Project
from datm.data_tools.source_gen.dataset_source import DatasetSource


@click.group()
def cmd_group():
    """
    Placeholder for 'group' of commands. The console script entry
    point ('datm'), as defined in the package's setup.py, is
    set to be this function. The available commands are defined
    below.

    """
    pass



@cmd_group.command()
@click.option('--addrport', default=None,
              help="Override the default ip address (optional) and port. Uses format: <ipaddr:port>")
def run(addrport):
    """
    Runs the datm server.
    """
    make_django_env()
    datm_ascii = """
====================================
 _____         _______
|  __ \    /\ |__   __|
| |  | |  /  \   | | _ __ ___
| |  | | / /\ \  | || '_ ` _ \\
| |__| |/ ____ \ | || | | | | |
|_____//_/    \_\|_||_| |_| |_|

===================================
***       SERVER RUNNING        ***
-----------------------------------

    """

    print datm_ascii
    threads = []

    # management.call_command('runserver', addrport=addrport, use_reloader=False, verbosity=0)

    t = threading.Thread(target=management.call_command, args=('runserver',), kwargs={'addrport': addrport,
                                                                                      'use_reloader': False,
                                                                                      'verbosity': 0})
    threads.append(t)
    t.start()
    if addrport is not None:
        webbrowser.open(addrport)
    else:
        webbrowser.open("http://127.0.0.1:8000/")


@cmd_group.command()
@click.argument('project_id', nargs=1)
@click.argument('dataset_id', nargs=1)
def generatesource(project_id, dataset_id):
    project = Project.objects.get(id=project_id)
    dataset = Dataset.objects.get(project_asset__id=dataset_id, project_asset__project=project)
    cwd = os.getcwd()
    source_str, immutable_dataframes = dataset.source()
    dataset_source = DatasetSource(dataset_name=dataset.name,
                                   immutable_dataframes=immutable_dataframes,
                                   source_str=source_str,
                                   dir_path=cwd)
    dataset_source.generate()

