from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

# Todo: remove unneeded URL patterns

from .views import Dashboard, NewProject, DeleteProject, DatmProject, NewDataset, NewTransformation, EditTransformation, \
    DatasetCSV, ViewDatasetCSV, VisualizationPNG, NewVisualization, EditVisualization, DeleteProjectAsset,\
    DatasetViewer, QuickCreateDataset, BakeDataset

urlpatterns = [
    # Dashboard
    url(r'^$', Dashboard.as_view(), name='home'),
    # New project
    url(r'^new/project/$', NewProject.as_view(), name='new_project'),
    # Datm project graph view
    url(r'^project/(?P<project_id>[\w-]+)/$', DatmProject.as_view(), name='project'),
    # Delete project.
    url(r'^delete/project/$', DeleteProject.as_view(), name='delete_project'),
    # Add project dataset.
    url(r'^project/(?P<project_id>[\w-]+)/add/dataset/$', NewDataset.as_view(), name='new_dataset'),
    # Create project dataset.
    url(r'^project/(?P<project_id>[\w-]+)/create/dataset/$', QuickCreateDataset.as_view(), name='create_dataset'),
    # Bake project dataset.
    url(r'^project/(?P<project_id>[\w-]+)/bake/dataset/$', BakeDataset.as_view(), name='bake_dataset'),
    # Delete project asset.
    url(r'^project/(?P<project_id>[\w-]+)/delete/asset/$', DeleteProjectAsset.as_view(), name='delete_project_asset'),
    # Add project dataset transformation.
    url(r'^project/(?P<project_id>[\w-]+)/add/transformation/$', NewTransformation.as_view(), name='new_transformation'),
    # New  visualization.
    url(r'^project/(?P<project_id>[\w-]+)/add/visualization/$', NewVisualization.as_view(), name='new_visualization'),
    # Edit transformation.
    url(r'^project/(?P<project_id>[\w-]+)/transformation/(?P<transformation_id>[\w-]+)/$', EditTransformation.as_view(), name='edit_transformation'),
    # Dataset Viewer
    url(r'^project/(?P<project_id>[\w-]+)/dataset/(?P<dataset_id>[\w-]+)/view/$', DatasetViewer.as_view(), name='view_dataset'),
    # Dataset CSV
    url(r'^project/(?P<project_id>[\w-]+)/dataset/(?P<dataset_id>[\w-]+)/csv/$', DatasetCSV.as_view(), name='dataset_csv'),
    # Dataset CSV viewer
    url(r'^project/(?P<project_id>[\w-]+)/dataset/(?P<dataset_id>[\w-]+)/csv/view/$', ViewDatasetCSV.as_view(), name='view_csv'),
    # Visualization PNG
    url(r'^project/(?P<project_id>[\w-]+)/visualization/(?P<visualization_id>[\w-]+)/plot\.png$', VisualizationPNG.as_view(), name='visualization_png'),
    # Visualization Editor
    url(r'^project/(?P<project_id>[\w-]+)/visualization/(?P<visualization_id>[\w-]+)/$', EditVisualization.as_view(), name='edit_visualization'),
]
