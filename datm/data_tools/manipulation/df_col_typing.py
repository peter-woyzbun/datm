import pandas as pd


class ColumnDTypeSet(object):

    def __init__(self, df=None, column_dtype_dict=None):
        self.df = df
        self.column_dtype_dict = column_dtype_dict
        if self.df is not None:
            self._create_column_dtype_dict()

    def _create_column_dtype_dict(self):
        self.column_dtype_dict = dict()
        for col in self.df.columns:
            self.column_dtype_dict[col] = str(self.df[col].dtype)

    def __sub__(self, other):
        """
        Override default subtraction behaviour to calculate the 'set difference'
        between this ColumnDType set instance and 'other' ColumnDType set instance.

        The set difference operation has the following rules:

            (1) Any columns in both sets with the same data types are
                removed.
            (2) Any columns in both sets with different data types are
                kept.
            (3) Any columns in this set and not the other are kept.

        """
        if isinstance(other, ColumnDTypeSet):
            type_diff_dict = dict()
            for k, v in self.column_dtype_dict.items():
                if k in other.column_dtype_dict:
                    if other.column_dtype_dict[k] != v:
                        type_diff_dict[k] = v
                else:
                    type_diff_dict[k] = v
            self.column_dtype_dict = type_diff_dict
        return self

    def apply_to_df(self, df):
        for k, v in self.column_dtype_dict.items():
            df[k] = df[k].astype(v)
        return df


df_1 = pd.DataFrame({'one': pd.Series([1., 2., 3., 5.]),
                   'two': pd.Series([1., 2., 3., 4.]),
                   'three': pd.Series(['cat1', 'cat1', 'cat2', 'cat3'], dtype="category")})

df_2 = pd.DataFrame({'one': pd.Series([1., 2., 3., 5.]),
                   'two': pd.Series([1., 2., 3., 4.], dtype="category"),
                   'three': pd.Series(['cat1', 'cat1', 'cat2', 'cat3'], dtype="category")})

df_1_type_set = ColumnDTypeSet(df=df_1)
df_2_type_set = ColumnDTypeSet(df=df_2)

type_diff = df_2_type_set - df_1_type_set

print(type_diff.column_dtype_dict)