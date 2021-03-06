from django.conf.urls import url, include
from django.conf import settings
from django.conf.urls.static import static

from .views import Dashboard, NewProject, DeleteProject, DatmProject, NewDataset, NewTransformation, EditTransformation, \
    DatasetCSV, ViewDatasetCSV, VisualizationPNG, NewVisualization, EditVisualization, DeleteProjectAsset,\
    DatasetViewer, QuickCreateDataset, BakeDataset, VisualizationImage


urlpatterns = [
    # Dashboard
    url(r'^$', Dashboard.as_view(), name='home'),
    # New project
    url(r'^new/project/$', NewProject.as_view(), name='new_project'),
    # Delete project.
    url(r'^delete/project/$', DeleteProject.as_view(), name='delete_project'),
    # Project URLs.
    url(r'^project/(?P<project_id>[\w-]+)/', include([
        # Project graph.
        url(r'^$', DatmProject.as_view(), name='project'),
        # Quick create dataset.
        url(r'^create/dataset/$', QuickCreateDataset.as_view(), name='create_dataset'),
        # Bake create dataset.
        url(r'^bake/dataset/$', BakeDataset.as_view(), name='bake_dataset'),
        # Delete asset.
        url(r'^delete/asset/$', DeleteProjectAsset.as_view(), name='delete_project_asset'),
        # Project asset creation.
        url(r'^add/', include([
            url(r'^dataset/$', NewDataset.as_view(), name='new_dataset'),
            url(r'^transformation/$', NewTransformation.as_view(), name='new_transformation'),
            url(r'^visualization/$', NewVisualization.as_view(), name='new_visualization'),
        ])),
        # Dataset views.
        url(r'^dataset/(?P<dataset_id>[\w-]+)/', include([
            url(r'^view/$', DatasetViewer.as_view(), name='view_dataset'),
            url(r'^csv/$', DatasetCSV.as_view(), name='dataset_csv'),
            url(r'^view/csv/$', ViewDatasetCSV.as_view(), name='view_csv'),
        ])),
        # Visualization views.
        url(r'^visualization/(?P<visualization_id>[\w-]+)/', include([
            url(r'^$', EditVisualization.as_view(), name='edit_visualization'),
            url(r'^image/$', VisualizationImage.as_view(), name='visualization_image'),
            url(r'^plot\.png$', VisualizationPNG.as_view(), name='visualization_png'),
        ])),
        # Edit transformation.
        url(r'^transformation/(?P<transformation_id>[\w-]+)/$', EditTransformation.as_view(),
            name='edit_transformation'),
    ])),
]