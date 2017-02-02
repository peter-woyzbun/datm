import os
import mimetypes
import json
import pandas as pd
from StringIO import StringIO

from django import http
from django.shortcuts import render, redirect
from django.views import View
from django.core.urlresolvers import reverse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.utils.http import http_date
from django.http import JsonResponse, HttpResponse

from .models import Project, ProjectAsset, Transformation, Dataset

from datm.data_tools.visualization.histogram import Histogram
from datm.web import USER_DATASET_PATH


class Dashboard(View):
    def get(self, request):
        projects = Project.objects.all()
        return render(request, 'core/dashboard.html', context={'projects': projects})


class NewProject(View):

    @csrf_exempt
    def post(self, request):
        project_name = request.POST.get("project_name")
        project_description = request.POST.get("project_description")
        new_project = Project.objects.create(name=project_name, description=project_description)
        new_project.save()
        return redirect(reverse('core:project', kwargs={'project_id': new_project.id}))


class DeleteProject(View):

    @csrf_exempt
    def post(self, request):
        project_id = request.POST.get("project_id")
        project = Project.objects.get(id=project_id)
        project.delete()
        return redirect(reverse('core:home'))


@method_decorator(csrf_exempt, name='dispatch')
class DeleteProjectAsset(View):

    def post(self, request, project_id):
        project = Project.objects.get(id=project_id)
        project_asset_id = request.POST.get("asset_id")
        project.graph.delete_asset_tree(parent_node_asset_id=project_asset_id)
        return redirect(reverse('core:project', kwargs={'project_id': project.id}))


class DatmProject(View):

    def get(self, request, project_id):
        project = Project.objects.get(id=project_id)
        return render(request, 'core/project.html', context={'project': project})

    def post(self, request):
        pass


# =================================================
# TRANSFORMATION VIEWS
# -------------------------------------------------

@method_decorator(csrf_exempt, name='dispatch')
class NewTransformation(View):

    def post(self, request, project_id):
        project = Project.objects.get(id=project_id)
        parent_dataset_id = request.POST.get("dataset_id")
        description = request.POST.get("description")
        child_dataset_name = request.POST.get("child_dataset_name")
        child_dataset_description = request.POST.get("child_dataset_description")
        transformation_type = request.POST.get("transformation_type")
        ProjectAsset.create_transformation(project=project,
                                           description=description,
                                           transform_type=transformation_type,
                                           parent_dataset_id=parent_dataset_id,
                                           child_dataset_name=child_dataset_name,
                                           child_dataset_description=child_dataset_description)
        return redirect(reverse('core:project', kwargs={'project_id': project.id}))


@method_decorator(csrf_exempt, name='dispatch')
class EditTransformation(View):

    def get(self, request, project_id, transformation_id):
        project = Project.objects.get(id=project_id)
        project_asset = ProjectAsset.objects.get(id=transformation_id)
        transformation = Transformation.objects.get(project_asset__project=project,
                                                    project_asset=project_asset)
        if transformation.type == 'sql':
            template = 'core/sql_query.html'
        else:
            template = 'core/manipulation_set.html'

        return render(request, template, context={'transformation': transformation,
                                                  'project': project})

    def post(self, request, project_id, transformation_id):
        project = Project.objects.get(id=project_id)
        project_asset = ProjectAsset.objects.get(id=transformation_id)
        transformation = Transformation.objects.get(project_asset__project=project,
                                                    project_asset=project_asset)
        transformation_json = request.POST.get("transformation_json")
        if transformation.type == 'manipulation_set':
            transformation.manipulation_set = transformation_json
            transformation.save()
        elif transformation.type == 'sql':
            print json.loads(transformation_json)['query']
            transformation.sql_query = json.loads(transformation_json)['query']
            transformation.save()
        error_data = transformation.execute()
        return JsonResponse(error_data, safe=False)
        # return redirect(reverse('core:edit_transformation', kwargs={'project_id': project_id}))

    def execute_manipulation_set(self, request, transformation):
        transformation_json = request.POST.get("transformation_json")
        transformation.manipulation_set = transformation_json
        transformation.save()
        error_data = transformation.execute()
        return JsonResponse(error_data)


# =================================================
# DATASET VIEWS
# -------------------------------------------------

@method_decorator(csrf_exempt, name='dispatch')
class NewDataset(View):

    def post(self, request, project_id):
        project = Project.objects.get(id=int(project_id))
        dataset_name = request.POST.get("dataset_name")
        description = request.POST.get("description")
        csv_file = request.FILES['datafile']
        df = pd.read_csv(csv_file)
        ProjectAsset.create_dataset(project=project,
                                    name=dataset_name,
                                    description=description,
                                    df=df,
                                    immutable=True)
        return redirect(reverse('core:project', kwargs={'project_id': project.id}))


@method_decorator(csrf_exempt, name='dispatch')
class QuickCreateDataset(View):

    def post(self, request, project_id):
        project = Project.objects.get(id=int(project_id))
        dataset_name = request.POST.get("dataset_name")
        description = request.POST.get("description")
        csv_data = request.POST.get("csv_data")
        df = pd.read_csv(StringIO(csv_data))

        ProjectAsset.create_dataset(project=project,
                                    name=dataset_name,
                                    description=description,
                                    df=df,
                                    immutable=True)
        return redirect(reverse('core:project', kwargs={'project_id': project.id}))


@method_decorator(csrf_exempt, name='dispatch')
class BakeDataset(View):

    def post(self, request, project_id):
        project = Project.objects.get(id=project_id)
        source_dataset_id = request.POST.get("source_dataset_id")
        dataset_name = request.POST.get("dataset_name")
        description = request.POST.get("description")

        dataset_asset = ProjectAsset.objects.get(id=int(source_dataset_id))

        ProjectAsset.create_dataset(project=project,
                                    name=dataset_name,
                                    description=description,
                                    df=dataset_asset.dataset.df,
                                    immutable=True)

        return redirect(reverse('core:project', kwargs={'project_id': project.id}))


class DatasetCSV(View):

    def get(self, request, project_id, dataset_id):
        dataset_asset = ProjectAsset.objects.get(id=dataset_id)
        dataset = dataset_asset.dataset
        fullpath = dataset.csv.path
        statobj = os.stat(fullpath)
        content_type, encoding = mimetypes.guess_type(fullpath)
        content_type = content_type or 'application/octet-stream'
        response = http.HttpResponse(open(fullpath, 'rb').read(), content_type=content_type)
        response["Last-Modified"] = http_date(statobj.st_mtime)
        response["Content-Length"] = statobj.st_size
        response['Content-Disposition'] = 'attachment; filename="%s.csv"' \
                                          % dataset_asset.name
        if encoding:
            response["Content-Encoding"] = encoding
        return response


class ViewDatasetCSV(View):

    def get(self, request, project_id, dataset_id):
        project = Project.objects.get(id=project_id)
        dataset_asset = ProjectAsset.objects.get(id=dataset_id)
        dataset = dataset_asset.dataset
        return render(request, 'core/csv_table.html', context={'project': project,
                                                               'dataset_asset': dataset,
                                                               'column_names': dataset.column_names,
                                                               'rows': dataset.preview_rows})


class DatasetViewer(View):

    def get(self, request, project_id, dataset_id):
        start_row = int(request.GET.get('start_row', 0))
        end_row = start_row + 20
        dataset_asset = ProjectAsset.objects.get(id=dataset_id)
        df = dataset_asset.dataset.subset_df(start_row=start_row, n_rows=20)
        columns_names = df.columns.values.tolist()
        rows = df.values.tolist()

        return render(request, 'core/dataset_viewer.html', context={'dataset_asset': dataset_asset,
                                                                    'start_row': start_row,
                                                                    'end_row': end_row,
                                                                    'column_names': columns_names,
                                                                    'rows': rows})


# =================================================
# VISUALIZATION VIEWS
# -------------------------------------------------

class VisualizationPNG(View):

    def get(self, request, project_id, visualization_id):
        visualization_asset = ProjectAsset.objects.get(id=visualization_id)
        response = HttpResponse(content_type='image/png')
        visualization_asset.visualization.print_to_response(response=response)
        return response


class NewVisualization(View):

    def post(self, request, project_id):
        project = Project.objects.get(id=project_id)
        dataset_id = request.POST.get("dataset_id")
        visualization_title = request.POST.get("visualization_title")
        visualization_type = request.POST.get("visualization_type")
        visualization = ProjectAsset.create_visualization(project=project, title=visualization_title,
                                                          visualization_type=visualization_type,
                                                          dataset_id=dataset_id)
        return redirect(reverse('core:edit_visualization', kwargs={'project_id': project.id,
                                                                   'visualization_id': visualization.id}))


class EditVisualization(View):

    def get(self, request, project_id, visualization_id):
        visualization_asset = ProjectAsset.objects.get(id=visualization_id)
        project = Project.objects.get(id=project_id)
        return render(request, 'core/visualization.html', context={'project': project,
                                                                   'visualization_asset': visualization_asset})

    def post(self, request, project_id, visualization_id):
        visualization_asset = ProjectAsset.objects.get(id=visualization_id)
        parameter_json = request.POST.get("parameter_json")
        visualization_asset.visualization.parameters = parameter_json
        visualization_asset.visualization.ready = True
        visualization_asset.visualization.save()

