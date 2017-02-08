from datm.data_tools.visualization.base import BaseVisualization
import seaborn as sns


class ViolinPlot(BaseVisualization):

    def __init__(self, title, x_label, y_label, df_column, orientation, categorical_col=None):
        self.df_column = df_column
        if categorical_col == 'None':
            self.categorical_col = None
        else:
            self.categorical_col = categorical_col
        self.orientation = orientation
        super(ViolinPlot, self).__init__(title=title, x_label=x_label, y_label=y_label)

    def _create_figure(self, df):
        if self.categorical_col is None:
            sns.violinplot(x=self.df_column, data=df, ax=self.subfigure)
        else:
            df[self.categorical_col] = df[self.categorical_col].astype('category')
            sns.violinplot(x=self.df_column, data=df, ax=self.subfigure, y=self.categorical_col, orient=self.orientation)