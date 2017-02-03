import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from abc import ABCMeta, abstractmethod


class BaseVisualization(object):

    def __init__(self, title, x_label, y_label):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
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

    @staticmethod
    def apply_common_style(figure):
        rcParams['font.family'] = 'serif'
        figure.patch.set_facecolor('white')

    def _create_labels(self):
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
