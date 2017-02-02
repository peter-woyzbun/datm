from datm.data_tools.visualization.base import BaseVisualization
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class Boxplot(BaseVisualization):

    def __init__(self, title, x_lab, y_lab, df_column, categorical_var=None):
        self.df_column = df_column
        self.categorical_var = categorical_var
        super(Boxplot, self).__init__(title=title, x_lab=x_lab, y_lab=y_lab)

    def _create_figure(self, df):
        if self.categorical_var is None:
            sns.boxplot(x=self.df_column, data=df, ax=self.subfigure)
        else:
            df[self.categorical_var] = df[self.categorical_var].astype('category')
            sns.boxplot(x=self.df_column, data=df, ax=self.subfigure, y=self.categorical_var)