from datm.data_tools.visualization.base import BaseVisualization
import seaborn as sns


class StripPlot(BaseVisualization):

    def __init__(self, title, x_label, y_label, df_column, jitter=True, categorical_col=None, hue_col=None):
        self.df_column = df_column
        self.categorical_col = categorical_col
        self.hue_col = hue_col
        self.jitter = jitter
        super(StripPlot, self).__init__(title=title, x_label=x_label, y_label=y_label)

    def _create_figure(self, df):
        if self.categorical_col is not None:
            df[self.categorical_col] = df[self.categorical_col].astype('category')
        if self.hue_col is not None:
            df[self.hue_col] = df[self.hue_col].astype('category')
        sns.stripplot(x=self.df_column, y=self.categorical_col,
                      hue=self.hue_col, data=df, ax=self.subfigure, jitter=self.jitter)