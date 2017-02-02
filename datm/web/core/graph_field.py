from django.db import models
import networkx as nx
from networkx.readwrite import json_graph
import json


class GraphField(models.CharField):

    def __init__(self, *args, **kwargs):
        super(GraphField, self).__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection, context):
        try:
            graph_data = json.loads(value)
            g = json_graph.node_link_graph(graph_data, multigraph=False, directed=True)
            return g
        except:
            g = nx.DiGraph()
            return g

    def get_db_prep_value(self, value, connection, prepared=False):
        if isinstance(value, nx.DiGraph):
            graph_data = json_graph.node_link_data(value)
            graph_string = json.dumps(graph_data)
            return graph_string
        else:
            return ''