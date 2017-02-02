from django.db import models
import networkx as nx
from networkx.readwrite import json_graph
import json


class ListField(models.CharField):

    def __init__(self, *args, **kwargs):
        super(ListField, self).__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection, context):
        lis = value.split(",")
        return lis

    def get_db_prep_value(self, value, connection, prepared=False):
        return ",".join(value)


print "".split(",")