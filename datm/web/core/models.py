from __future__ import unicode_literals

import json
import os
import textwrap

import networkx as nx
import pandas as pd

import datm.utils
import datm.data_tools.source_gen.templates as source_templates
from datm.data_tools.transformations.manipulation_sets.manipulation_set import ManipulationSet
from datm.data_tools.transformations.sql.sql_query import SqlQuery
from datm.data_tools.visualization import Histogram, Boxplot, ViolinPlot, StripPlot, SwarmPlot
from datm.utils.func_timer import timeit
from datm.web import USER_DATASET_PATH

from django.db import models
from django.template import loader, Context, Template
from django.dispatch import receiver
from django.db.models.signals import post_save, pre_save, pre_delete

from model_fields import ListField, DictField
from .graph_field import GraphField


# =============================================
# Project Model
# ---------------------------------------------

class Project(models.Model):
    """
    Project model.

    Fields
    ------
    name : CharField
        Name given to the project.
    description : TextField
        Description of project. For use by user to identify project.
    created_on : DateField
        Date the project was created.

    Signals
    -------
    post_save: Create a Graph instance upon first save.

    """
    name = models.CharField(max_length=200)
    description = models.TextField(default='')
    created_on = models.DateField(auto_now_add=True, blank=True)


# ---------------------------------------------
# Project Model Signals
# ---------------------------------------------

@receiver(post_save, sender=Project)
def create_project_graph(sender, **kwargs):
    """
    Create Project graph on new Project creation.

    """
    newly_created = kwargs.get('created', False)
    if newly_created:
        project = kwargs.get('instance')
        project_graph = Graph.objects.create(project=project)
        project_graph.save()


# =============================================
# Graph Model
# ---------------------------------------------

class Graph(models.Model):
    """
    Graph model - captures the relationships between all ProjectAssets
    associated with a project.

    Fields
    ------
    project : OneToOneField
        The associated project.
    _graph : GraphField
        Custom field that saves/retrieves a Networkx directed graph. The field
        is defined in model_fields/graph_field.py.

    """
    project = models.OneToOneField(Project, on_delete=models.CASCADE, primary_key=True)
    data = models.CharField(max_length=10000, default='_')
    _graph = GraphField(max_length=10000, default='')

    def add_node(self, name, asset_id, type):
        """
        Add a node/project asset to the project graph.

        Parameters
        ----------
        name : str
            Name of node to add. Used for labelling graph visualization/plot.
        asset_id : int
            The ID of the ProjectAsset the node represents.
        type : str
            The type of ProjectAsset the node represents - 'dataset' or 'transformation'.

        """
        self._graph.add_node(asset_id, type=type, asset_id=asset_id, name=name)
        self.save()

    def add_edge(self, from_node_asset_id, to_node_asset_id, edge_type):
        """
        Add an edge between two nodes/project assets in the project graph.

        Parameters
        ----------
        from_node_asset_id : int
            The ID of the ProjectAsset of the edge's origin node.
        to_node_asset_id : int
            The ID of the ProjectAsset of the edge's destination node.
        edge_type : str
            The type of edge - 'transformation' or 'join'.

        """
        self._graph.add_edge(from_node_asset_id, to_node_asset_id)
        self._graph[from_node_asset_id][to_node_asset_id]['type'] = edge_type
        self.save()

    def remove_edge(self, from_node_asset_id, to_node_asset_id):
        """
        Remove the edge between two given nodes.

        Parameters
        ----------
        from_node_asset_id : int
            The ID of the ProjectAsset of the edge's origin node.
        to_node_asset_id : int
            The ID of the ProjectAsset of the edge's destination node.

        """
        # G = self.retrieve()
        self._graph.remove_edge(from_node_asset_id, to_node_asset_id)
        # G_data = json_graph.node_link_data(G)
        # G_json = json.dumps(G_data)
        # self.data = G_json
        self.save()

    def has_edge(self, from_node_asset_id, to_node_asset_id):
        return self._graph.has_edge(from_node_asset_id, to_node_asset_id)

    @property
    def graph_viz_data(self, origin_node_asset_id=None):

        node_type_templates = {
            "dataset": loader.get_template('core/project_graph/node_labels/dataset.html'),
            "manipulation_set": loader.get_template('core/project_graph/node_labels/transformation.html'),
            "sql": loader.get_template('core/project_graph/node_labels/sql_query.html'),
            "visualization": loader.get_template('core/project_graph/node_labels/visualization.html')
        }

        G = self._graph
        # If origin_node_asset_id is given, only include its dependents - i.e. create a subgraph.
        if origin_node_asset_id is not None:
            nodes = [origin_node_asset_id] + self.node_dependents(node_asset_id=origin_node_asset_id)
        else:
            nodes = nx.nodes(G)
        edge_tuples = nx.edges(G, nbunch=nodes)
        viz_data = {'nodes': [], 'edges': []}
        for node in nodes:
            project_asset = ProjectAsset.objects.get(id=node)
            if project_asset.type != 'transformation':
                label_template = node_type_templates[project_asset.type]
            else:
                label_template = node_type_templates[project_asset.transformation.type]
            context = Context({project_asset.type: project_asset.as_type,
                               'project': self.project})
            node_label = label_template.render(context)
            node_data = {'id': node, 'class': '%s-node' % project_asset.type, 'label': ''.join(node_label.splitlines())}
            viz_data['nodes'].append(node_data)
        for edge in edge_tuples:
            viz_data['edges'].append({'parent_node_id': edge[0],
                                      'child_node_id': edge[1],
                                      'class': '%s-edge' % G[edge[0]][edge[1]]['type']})

        return viz_data

    def dagre_js(self):
        template = loader.get_template('core/project_graph/graph.html')
        dagre_data = self.graph_viz_data
        context = Context({'nodes': dagre_data['nodes'],
                           'edges': dagre_data['edges']})

        return template.render(context)

    def transformation_tree(self, transformation_asset_id):
        """ Get all transformations through which a parent transformation must propogate. """
        G = self._graph
        transformation_list = []
        # Run depth-first search (traversal) of graph to get ordered transformation list.
        for node in list(nx.dfs_preorder_nodes(G, source=transformation_asset_id)):
            if G.node[node]['type'] == 'transformation':
                transformation_list.append(G.node[node]['asset_id'])
        return transformation_list

    def child_node_tree(self, parent_node_asset_id, target_asset_type=None):
        child_node_asset_ids = list()
        G = self._graph
        for node in list(nx.dfs_preorder_nodes(G, source=parent_node_asset_id)):
            if target_asset_type:
                if G.node[node]['type'] == target_asset_type:
                    child_node_asset_ids.append(G.node[node]['asset_id'])
            else:
                child_node_asset_ids.append(G.node[node]['asset_id'])
        return child_node_asset_ids

    def node_predecessors(self, asset_id, target_asset_type=None):
        G = self._graph
        predecessor_list = G.predecessors(asset_id)
        if target_asset_type:
            filtered_predecessor_list = list()
            for node in predecessor_list:
                if G.node[node]['type'] == target_asset_type:
                    filtered_predecessor_list.append(node)
            return filtered_predecessor_list
        else:
            return predecessor_list

    def transformation_successor_batches(self, transformation_asset_id):
        G = self._graph
        transformation_nodes = self.transformation_tree(transformation_asset_id=transformation_asset_id)
        batches = list()
        for node in transformation_nodes:
            if not batches:
                batches.append([node])
            else:
                last_node = batches[-1][-1]
                if node in G.successors(last_node):
                    batches.append([node])
                else:
                    batches[-1].append(node)
        return batches

    def node_dependents(self, node_asset_id, reverse=False):
        G = self._graph
        dfs = nx.dfs_successors(G, source=node_asset_id)
        topological_ordering = sorted({x for v in dfs.itervalues() for x in v})
        if reverse:
            topological_ordering = topological_ordering[::-1]
        return topological_ordering

    def node_ancestors(self, node_asset_id, keep_type=None):
        """
        Returns all ancestors of given node in topological order. If
        a 'keep_type' is given, only nodes of that type are returned.

        Returns: list of node asset ids.

        """
        ancestor_ids = nx.ancestors(self._graph, node_asset_id)
        topological_ordering = nx.topological_sort(self._graph, ancestor_ids)
        if keep_type is not None:
            filtered_top_ord = list()
            for node in topological_ordering:
                if self._graph.node[node]['type'] == keep_type:
                    filtered_top_ord.append(node)
            return filtered_top_ord
        else:
            return topological_ordering

    def transformation_joined_datasets(self, transformation_asset_id):
        G = self._graph
        predecessors = self.node_predecessors(asset_id=transformation_asset_id)
        joined_dataset_ids = list()
        for node in predecessors:
            if G.edge[node][transformation_asset_id]['type'] == 'join':
                joined_dataset_ids.append(node)
        return joined_dataset_ids

    def remove_nodes(self, node_asset_ids):
        for node_asset_id in node_asset_ids:
            self._graph.remove_node(node_asset_id)
        self.save()

    def delete_asset_tree(self, parent_node_asset_id):
        parent_node_asset_id = int(parent_node_asset_id)
        # Get any/all child nodes in topological order.
        child_nodes = self.node_dependents(node_asset_id=parent_node_asset_id, reverse=True)
        # Iterate over the child nodes in reverse topological order - we
        # want to delete nodes with no child nodes first, otherwise we
        # might 'break' foreign-key relations.
        for child_node_asset_id in child_nodes:
            project_asset = ProjectAsset.objects.get(id=child_node_asset_id)
            project_asset.delete()
        parent_project_asset = ProjectAsset.objects.get(id=parent_node_asset_id)
        parent_project_asset.delete()
        child_nodes.append(parent_node_asset_id)
        self.remove_nodes(node_asset_ids=child_nodes)


# =============================================
# Project Asset Model
# ---------------------------------------------

class ProjectAsset(models.Model):
    """
    ProjectAsset model - each project asset is a dataset, transformation,
    or visualization, associated with its project. Each asset also has an
    associated node defined in the associated project's graph.

    Fields
    ------
    project : OneToOneField
        The associated project.
    name : CharField
        The name given to the asset by the user.
    type : CharField
        The type of asset: 'transformation', 'dataset', or 'visualization'.
    description : CharField
        Description of the asset, as defined by the user.

    Signals
    -------
    post_save: Create a graph node on first save.

    """
    project = models.ForeignKey(Project, default=1, related_name='asset_set')
    name = models.CharField(max_length=200, default='Asset Name')
    type = models.CharField(max_length=200, default='dataset')
    description = models.CharField(max_length=1000, default='Project asset description.')

    @property
    def description_html(self):
        # Use textwrap because the graph visualization CSS isn't behaving.
        return "<br>".join(textwrap.wrap(self.description, width=20))

    @property
    def as_type(self):
        return getattr(self, self.type)

    @classmethod
    def create_dataset(cls, project, name, description, df, immutable=False):
        """
        Create a new project Dataset. This requires:

            (1) Creating and saving the related ProjectAsset.
            (2) Creating and saving the actual Dataset, and saving the dataframe as an HDF5 file.
            (3) Adding the dataset node to the project graph - done via post-save signal.

        Parameters
        ----------
        project : Project
            Project instance associated with dataset.
        name : str
            Name of new dataset.
        description : str
            Description of new dataset.
        df : Pandas DataFrame
            DataFrame containing dataset data.
        immutable : bool
            Whether or not the dataset is 'immutable'. Immutable datasets are
            not the product of a transformation.

        """
        project_asset = cls.objects.create(project=project, name=name, description=description, type="dataset")
        project_asset.save()
        dataset = Dataset.objects.create(project_asset=project_asset, immutable=immutable)
        dataset.save()
        dataset.save_df_to_hdf(df)

    @classmethod
    def create_transformation(cls, project, description, transform_type, parent_dataset_id,
                              child_dataset_name, child_dataset_description):
        """
        Create a new transformation. This requires:

            (1) Creating and saving the related ProjectAsset.
            (2) Creating and saving the child Dataset.
            (3) Creating and saving the actual transformation.

        Parameters
        ----------
        project : Project
            Project instance associated with dataset.
        description : str
            Description of new transformation.
        transform_type : str
            The type of transformation - 'manipulation_set' or 'sql'.
        parent_dataset_id : int
            The ID of the parent Dataset.
        child_dataset_name : str
            Name given to child Dataset.
        child_dataset_description : str
            Description given to child Dataset.

        """

        # Create child dataset first.
        parent_dataset = Dataset.objects.get(project_asset=parent_dataset_id)
        child_dataset_asset = cls.objects.create(project=project,
                                                 name=child_dataset_name,
                                                 description=child_dataset_description,
                                                 type="dataset")
        child_dataset_asset.save()
        child_dataset = Dataset.objects.create(project_asset=child_dataset_asset)
        child_dataset.save()
        child_dataset.save_df_to_hdf(df=parent_dataset.df)
        transformation_asset = cls.objects.create(project=project,
                                                  type="transformation",
                                                  description=description)
        transformation_asset.save()
        transformation = Transformation.objects.create(project_asset=transformation_asset,
                                                       parent_dataset=parent_dataset,
                                                       child_dataset=child_dataset,
                                                       type=transform_type)
        transformation.save()

    @classmethod
    def create_visualization(cls, project, title, visualization_type, dataset_id):
        """
        Create a new visualization. This requires:

            (1) Creating and saving the related ProjectAsset.
            (2) Creating and saving the actual visualization.

        Parameters
        ----------
        project : Project
            Project instance associated with dataset.
        title : str
            Title of the new visualization
        visualization_type : str
            The new visualization's type (e.g. 'boxplot', or 'histogram'.
        dataset_id : int
            The ID of the associated parent Dataset.

        Returns
        -------
        visualization : Visualization
            Newly created Visualization instance.

        """
        visualization_asset = cls.objects.create(name=title, type="visualization", project=project)
        visualization_asset.save()
        dataset_asset = ProjectAsset.objects.get(id=dataset_id)
        visualization = Visualization(project_asset=visualization_asset,
                                      dataset=dataset_asset.dataset,
                                      type=visualization_type)
        visualization.save()
        return visualization


# ---------------------------------------------
# Project Asset Signals
# ---------------------------------------------

@receiver(post_save, sender=ProjectAsset)
def create_project_asset_node(sender, **kwargs):
    """
    Create graph node upon ProjectAsset creation.

    """
    newly_created = kwargs.get('created', False)
    if newly_created:
        project_asset = kwargs.get('instance')
        project_asset.project.graph.add_node(name=project_asset.name,
                                             asset_id=project_asset.id,
                                             type=project_asset.type)


# =============================================
# Dataset Model
# ---------------------------------------------

class Dataset(models.Model):
    """
    Dataset model - represents a dataframe.

    Fields
    ------
    project_asset : OneToOneField
        The associated ProjectAsset.
    hdf : FileField
        The HDF5 file containing the dataframe.
    created_at : DateTimeField
        The datetime the dataset was added/created.
    columns : ListField
        Custom field that saves/retrieves a list of dataframe columns.
        The field is defined in model_fields/list_field.py.
    n_rows : IntegerField
        The number of rows contained in the dataset's dataframe.
    immutable : Boolean
        Indicates whether or not the dataset is the result of a
        transformation.
    column_dtypes : DictField
        Custom field that saves/retrieves a dictionary as JSON that maps
        column names to their data types. This allows type information
        to be preserved and no type information to be 'lost' during the
        read/write process. The field is defined in model_fields/list_field.py.

    Signals
    -------
    pre_delete : When a Dataset is deleted, delete its associated HDF5 file.

    """
    project_asset = models.OneToOneField(ProjectAsset, on_delete=models.CASCADE, primary_key=True)
    csv = models.FileField(default='something', upload_to='user_datasets')
    hdf = models.FileField(default='something', upload_to=USER_DATASET_PATH)
    created_at = models.DateTimeField(auto_now_add=True)
    columns = ListField(default='_', max_length=5000)
    n_rows = models.IntegerField(default=0)
    immutable = models.BooleanField(default=False)
    column_dtypes = DictField(default="{}", max_length=5000)

    def save_df_to_hdf(self, df):
        """
        Save DataFrame to HDF format, first completing the following:

            (1) Ensure all DataFrame column names are 'valid'.
            (2) Record/save the column names.
            (3) Record the number of rows.

        Parameters
        ----------
        df : Pandas DataFrame
            DataFrame to save as HDF.

        """
        df = self._validate_col_names(df)
        self.columns = df.columns.values.tolist()
        self.n_rows = len(df.index)
        hdf_path = os.path.join(USER_DATASET_PATH, '%s_%s_%s.h5' % (self.project_asset.project.id, self.id, self.name))
        df.to_hdf(hdf_path, key=self.name, format='table')
        self.hdf.name = hdf_path
        self.column_dtypes = self.col_dtypes_dict(df=df)
        self.save()

    @staticmethod
    def _validate_col_names(df):
        """ Ensure that all dataframe columns are valid Python variable names. """
        for column in df.columns.values.tolist():
            df = df.rename(columns={column: datm.utils.clean_var_name(column)})
        return df

    @staticmethod
    def col_dtypes_dict(df):
        """ Return a dictionary mapping given dataframe columns to data types. """
        column_dtypes_dict = dict()
        for col in df.columns:
            column_dtypes_dict[col] = str(df[col].dtype)
        return column_dtypes_dict

    def ordered_col_dtypes(self):
        """ Return a list of column data types, in the order they appear in the DataFrame. """
        col_dtypes = list()
        for column in self.columns:
            col_dtypes.append(self.column_dtypes[column])
        return col_dtypes

    def apply_dtypes_to_df(self, df):
        for k, v in self.column_dtypes.items():
            df[k] = df[k].astype(v)
        return df

    def source_code(self):
        pass

    class Meta:
        app_label = 'core'

    @property
    def id(self):
        return self.project_asset.id

    @property
    def name(self):
        return self.project_asset.name

    @property
    def description(self):
        return self.project_asset.description

    @property
    def description_html(self):
        return self.project_asset.description_html

    @property
    def df(self):
        """ Return a pandas dataframe for the given dataset. """

        df = pd.read_hdf(self.hdf.path)
        if not self.immutable:
            df = self.apply_dtypes_to_df(df)
        return df

    @property
    def column_names(self):
        return self.column_dtypes.keys()

    def subset_df(self, start_row, n_rows):
        stop_row = int(start_row) + int(n_rows)
        df = pd.read_hdf(self.hdf.path, start=start_row, stop=stop_row)
        return df[start_row:stop_row]

    @property
    def hdf_path(self):
        return os.path.join(USER_DATASET_PATH, self.hdf.name)

    def source(self):
        source = ""
        immutable_dataframes = {self.name: self.hdf_path}

        ancestor_dataset_ids = self.project_asset.project.graph.node_ancestors(node_asset_id=self.id,
                                                                               keep_type='dataset')
        for dataset_id in ancestor_dataset_ids:
            dataset_asset = ProjectAsset.objects.get(id=dataset_id)
            source += dataset_asset.dataset.node_source()
            source += "\n"
            if dataset_asset.dataset.immutable:
                immutable_dataframes[dataset_asset.name] = dataset_asset.dataset.hdf_path
        return source, immutable_dataframes

    def node_source(self):
        if self.immutable:
            source_template = Template(source_templates.immutable_dataset)
            context = Context({'dataset_filename': "%s.csv" % self.name,
                               'dataset_name': self.name})
            source_str = source_template.render(context)
            return source_str
        else:
            parent_transformation = Transformation.objects.get(child_dataset=self)
            source_template = Template(source_templates.transformation_src)
            transformation_src = parent_transformation.node_source()
            context = Context({'parent_dataset_name': parent_transformation.parent_dataset.name,
                               'child_dataset_name': parent_transformation.child_dataset.name,
                               'transformation_src': transformation_src})
            source_str = source_template.render(context)
            return source_str


# ---------------------------------------------
# Dataset Signals
# ---------------------------------------------

@receiver(pre_delete, sender=Dataset)
def delete_dataset_files(sender, **kwargs):
    """
    Remove dataset files upon dataset deletion.

    """
    dataset = kwargs.get('instance')
    try:
        os.remove(dataset.hdf_path)
        print("Successfully deleted HDF file.")
    except:
        pass


# =============================================
# Transformation Model
# ---------------------------------------------

class Transformation(models.Model):
    project_asset = models.OneToOneField(ProjectAsset, on_delete=models.CASCADE, primary_key=True,
                                         related_name='transformation')
    parent_dataset = models.ForeignKey(Dataset, related_name='child_transformation')
    child_dataset = models.ForeignKey(Dataset, related_name='parent_transformation')
    created_at = models.DateTimeField(auto_now_add=True)
    manipulation_set = models.CharField(max_length=10000, default='')
    sql_query = models.CharField(max_length=10000, default='')
    type = models.CharField(max_length=10000, default='manipulation_set')
    has_errors = models.BooleanField(default=False)
    joined_datasets = models.ManyToManyField(ProjectAsset, related_name='child_joins')

    @property
    def id(self):
        return self.project_asset.id

    @property
    def name(self):
        return self.project_asset.name

    @property
    def description(self):
        return self.project_asset.description

    @property
    def description_html(self):
        return self.project_asset.description_html

    @property
    def manipulation_list(self):
        try:
            return json.loads(self.manipulation_set)['m']
        except:
            return None

    def _clear_existing_joins(self):
        """
        Remove any existing joins/edges between any dataset and this
        transformation from the graph.

        """
        for dataset in self.joined_datasets.all():
            if self.project_asset.project.graph.has_edge(from_node_asset_id=dataset.id, to_node_asset_id=self.id):
                self.project_asset.project.graph.remove_edge(from_node_asset_id=dataset.id,
                                                             to_node_asset_id=self.id)
        self.joined_datasets.clear()

    def _create_join_edges(self):
        for dataset in self.joined_datasets.all():
            self.project_asset.project.graph.add_edge(from_node_asset_id=dataset.id,
                                                      to_node_asset_id=self.id,
                                                      edge_type='join')

    @timeit
    def execute(self):
        """ Execute the transformation given its type. """
        if self.type == 'manipulation_set':
            return self._execute_manipulation_set()
        elif self.type == 'sql':
            return self._execute_sql_query()

    def _execute_manipulation_set(self):
        self._clear_existing_joins()
        manipulation_set = ManipulationSet.create_from_list(dataset_name=self.child_dataset.name,
                                                            manipulation_list=self.manipulation_list)
        # Create a dictionary mapping joined dataset ids to their dataframes.
        joined_df_dict = dict()
        for joined_dataset_id in manipulation_set.joined_dataset_ids:
            dataset_asset = ProjectAsset.objects.get(id=joined_dataset_id)
            joined_df_dict[dataset_asset.id] = dataset_asset.dataset.df
            self.joined_datasets.add(dataset_asset)
        manipulation_set.set_required_join_dfs(df_dict=joined_df_dict)
        df = self.parent_dataset.df
        transformed_df = manipulation_set.execute(df)
        self.child_dataset.save_df_to_hdf(df=transformed_df)
        self._create_join_edges()
        if not manipulation_set.execution_successful:
            self.has_errors = True
            self.save()
        elif manipulation_set.execution_successful:
            self.has_errors = False
            self.save()
        return manipulation_set.error_data

    def joinable_dataset_map(self):
        dataset_map = dict()
        for dataset in self.joinable_datasets:
            dataset_map[dataset.name] = dataset.id
        return dataset_map

    def _execute_sql_query(self):
        self._clear_existing_joins()
        sql_query = SqlQuery(query=self.sql_query, joinable_dataset_map=self.joinable_dataset_map())
        tables = {self.parent_dataset.name: self.parent_dataset.df}
        joined_dataset_assets = [ProjectAsset.objects.get(id=dataset_id) for dataset_id in sql_query.joined_dataset_ids]
        for dataset_asset in joined_dataset_assets:
            tables[dataset_asset.name] = dataset_asset.dataset.df
        try:
            df = sql_query.execute(tables=tables)
            self.child_dataset.save_df_to_hdf(df=df)
            if joined_dataset_assets:
                self.joined_datasets.add(joined_dataset_assets)
            return {'error': False}
        except:
            return {'error': True}

    @property
    def joinable_datasets(self):
        """
        Return list of Dataset assets that are joinable with...
        This is the set difference between all project dataset ids and the ids of
        those datasets that are 'reachable' from the transformation graph node.

        """
        project_dataset_ids = ProjectAsset.objects.filter(project=self.project_asset.project,
                                                          type='dataset').values_list('id', flat=True)
        child_dataset_ids = self.project_asset.project.graph.child_node_tree(parent_node_asset_id=self.project_asset.id,
                                                                             target_asset_type='dataset')
        joinable_dataset_ids = list(set(project_dataset_ids) - set(child_dataset_ids))
        joinable_dataset_list = list()

        for dataset_asset_id in joinable_dataset_ids:
            if dataset_asset_id != self.parent_dataset.id:
                dataset_asset = ProjectAsset.objects.get(id=dataset_asset_id)
                joinable_dataset_list.append(dataset_asset.dataset)
        return joinable_dataset_list

    def node_source(self):
        if self.type == 'manipulation_set':
            return self._manipulation_set_source()
        elif self.type == 'sql':
            return self._sql_source()

    def _manipulation_set_source(self):
        manipulation_set = ManipulationSet.create_from_list(dataset_name=self.child_dataset.name,
                                                            manipulation_list=self.manipulation_list,
                                                            source_code_mode=True)
        joined_df_dict = dict()
        for joined_dataset_id in manipulation_set.joined_dataset_ids:
            dataset_asset = ProjectAsset.objects.get(id=joined_dataset_id)
            joined_df_dict[dataset_asset.id] = dataset_asset.name
        manipulation_set.set_required_join_dfs(joined_df_dict)
        df = pd.read_hdf(self.parent_dataset.hdf_path)
        source_str = manipulation_set.execute(df)
        return source_str

    def _sql_source(self):
        sql_query = SqlQuery(query=self.sql_query,
                             joinable_dataset_map=self.joinable_dataset_map(),
                             source_code_mode=True)
        tables = {self.parent_dataset.name: "%s_df" % self.parent_dataset.name}
        joined_dataset_assets = [ProjectAsset.objects.get(id=dataset_id) for dataset_id in sql_query.joined_dataset_ids]
        for dataset_asset in joined_dataset_assets:
            tables[dataset_asset.name] = "%s_df" % dataset_asset.name
        source_str = "%s_df = %s" % (self.child_dataset.name, sql_query.execute(tables))
        return source_str


# ---------------------------------------------
# Transformation Signals
# ---------------------------------------------

@receiver(post_save, sender=Transformation)
def create_transformation_edges(sender, **kwargs):
    """
    Create transformation graph edges upon creation.
    """
    newly_created = kwargs.get('created', False)
    if newly_created:
        transformation = kwargs.get('instance')
        transformation.project_asset.project.graph.add_edge(from_node_asset_id=transformation.parent_dataset.id,
                                                            to_node_asset_id=transformation.id,
                                                            edge_type="normal")
        transformation.project_asset.project.graph.add_edge(from_node_asset_id=transformation.id,
                                                            to_node_asset_id=transformation.child_dataset.id,
                                                            edge_type="normal")


# =============================================
# Visualization Model
# ---------------------------------------------

class Visualization(models.Model):
    project_asset = models.OneToOneField(ProjectAsset, on_delete=models.CASCADE, primary_key=True,
                                         related_name='visualization')
    dataset = models.ForeignKey(Dataset, related_name='visualization_set')
    parameters = models.CharField(max_length=10000, default='')
    labels = DictField(default="{}", max_length=5000)
    options = DictField(default="{}", max_length=5000)
    type = models.CharField(max_length=100, default='')
    friendly_type = models.CharField(max_length=100, default='')
    ready = models.BooleanField(default=False)
    has_errors = models.BooleanField(default=False)

    @property
    def id(self):
        return self.project_asset.id

    @property
    def name(self):
        return self.project_asset.name

    def print_to_response(self, response):
        visualization_class_map = {'histogram': Histogram,
                                   'boxplot': Boxplot,
                                   'violin': ViolinPlot,
                                   'strip': StripPlot,
                                   'swarm': SwarmPlot}

        visualization = visualization_class_map[self.type](**dict(self.options.items() + self.labels.items()))
        visualization.create_figure(self.dataset.df)
        visualization.print_to_response(response=response)


# ---------------------------------------------
# Visualization Signals
# ---------------------------------------------

@receiver(post_save, sender=Visualization)
def create_visualization_edge(sender, **kwargs):
    """
    Create visualization graph edge upon creation.
    """
    newly_created = kwargs.get('created', False)
    if newly_created:
        visualization = kwargs.get('instance')
        visualization.project_asset.project.graph.add_edge(from_node_asset_id=visualization.dataset.id,
                                                           to_node_asset_id=visualization.id,
                                                           edge_type="normal")