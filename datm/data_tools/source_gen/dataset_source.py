import os
import autopep8
import pandas as pd


class DatasetSource(object):

    def __init__(self, dataset_name, immutable_dataframes, source_str, dir_path):
        self.dataset_name = dataset_name
        self.immutable_dataframes = immutable_dataframes
        self.source_str = source_str
        self.dir_path = dir_path

    def generate(self):
        self._create_data_folder()
        self._write_immutable_dataframes()
        self._add_imports()
        self._format_source_str()
        source_file = open("%s.py" % self.dataset_name, "w")
        source_file.write(self.source_str)
        source_file.close()

    def _add_imports(self):
        new_string = "import pandas as pd \n"
        new_string += "import numpy as np \n \n"
        new_string += self.source_str
        self.source_str = new_string

    def _format_source_str(self):
        print "Formatting source code."
        self.source_str = autopep8.fix_code(self.source_str)

    @property
    def data_dir_path(self):
        """ Path to folder containing CSV files. """
        return os.path.join(self.dir_path, 'data')

    def _create_data_folder(self):
        """ Create folder for storing immutable datasets (CSV's). """
        print "Creating data folder."
        if not os.path.isdir(self.data_dir_path):
            os.makedirs(self.data_dir_path)

    def dataset_csv_path(self, dataset_name):
        return os.path.join(self.data_dir_path, '%s.csv' % dataset_name)

    def _write_immutable_dataframes(self):
        print "Writing CSV files."
        for dataset_name, hdf_path in self.immutable_dataframes.items():
            df = pd.read_hdf(hdf_path)
            df.to_csv(self.dataset_csv_path(dataset_name=dataset_name), index=False)
