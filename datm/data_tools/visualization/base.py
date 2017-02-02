import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import rcParams


import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from abc import ABCMeta, abstractmethod


class BaseVisualization(object):

    def __init__(self, title, x_lab, y_lab):
        self.title = title
        self.x_lab = x_lab
        self.y_lab = y_lab
        self.figure = plt.figure()
        self.subfigure = self.figure.add_subplot(1, 1, 1)

    def print_to_response(self, response):
        self.apply_common_style(self.figure)
        canvas = FigureCanvas(self.figure)
        canvas.draw()
        canvas.print_png(response)

    def create_figure(self, df):
        self._create_figure(df)
        self._create_labels()

    @abstractmethod
    def _create_figure(self, df):
        raise NotImplementedError

    def apply_common_style(self, figure):
        rcParams['font.family'] = 'serif'
        figure.patch.set_facecolor('white')

    def _create_labels(self):
        plt.title(self.title)
        plt.xlabel(self.x_lab)
        plt.ylabel(self.y_lab)
