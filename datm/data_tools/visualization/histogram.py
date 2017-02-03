import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from datm.data_tools.visualization.base import BaseVisualization


class Histogram(BaseVisualization):

    def __init__(self, title, x_label, y_label, df_column, bins, normed=False):
        self.df_column = df_column
        self.bins = int(bins)
        self.normed = normed
        super(Histogram, self).__init__(title=title, x_label=x_label, y_label=y_label)

    def _create_figure(self, df):
        df = df.dropna()
        sns.distplot(df[self.df_column], ax=self.subfigure)

