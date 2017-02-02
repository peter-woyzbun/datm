from django.db import models
import networkx as nx
from networkx.readwrite import json_graph
import json


class DictField(models.CharField):

    def __init__(self, *args, **kwargs):
        super(DictField, self).__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection, context):
        try:
            return json.loads(value)
        except:
            return dict()

    def get_db_prep_value(self, value, connection, prepared=False):
        return json.dumps(value)