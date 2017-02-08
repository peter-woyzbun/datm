from datm.data_tools.visualization.base import BaseVisualization
import seaborn as sns


class Heatmap(BaseVisualization):

    def __init__(self, title, x_label, y_label, y_column, x_column, value_col, pivot=True):
        self.y_column = y_column
        self.x_column = x_column
        self.value_col = value_col
        self.pivot = pivot

        super(Heatmap, self).__init__(title=title, x_label=x_label, y_label=y_label)

    def _create_figure(self, df):
        if self.pivot:
            df = df.pivot(self.y_column, self.x_column, self.value_col)
        else:
            pass