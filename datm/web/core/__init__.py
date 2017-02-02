import os


def setup():
    module = os.path.split(os.path.dirname(__file__))[-1]
    # os.environ.setdefault("DJANGO_SETTINGS_MODULE", "{}.settings".format(module))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'datm.web.web.settings')
    import django
    django.setup()