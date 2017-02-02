import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from datm.data_tools.visualization.base import BaseVisualization


class Histogram(BaseVisualization):

    def __init__(self, title, x_lab, y_lab, df_column, bins, normed=False):
        self.title = title
        self.x_lab = x_lab
        self.y_lab = y_lab
        self.df_column = df_column
        self.bins = int(bins)
        self.normed = normed

    def create_figure(self, df):
        figure = plt.figure()
        subfigure = figure.add_subplot(1, 1, 1)
        sns.set_style("whitegrid")
        sns.despine(left=True)
        sns.distplot(df[self.df_column], ax=subfigure)
        # subfigure.hist(df[self.df_column], bins=self.bins, normed=self.normed)
        # subfigure.spines["top"].set_visible(False)
        # subfigure.spines["right"].set_visible(False)
        # subfigure.get_xaxis().tick_bottom()
        # subfigure.get_yaxis().tick_left()
        # Add labels.
        plt.title(self.title)
        plt.xlabel(self.x_lab)
        plt.ylabel(self.y_lab)
        return figure

