import pandas as pd
import numpy as np


# ======================================
# Custom Evaluator Functions

def rollmean(col, n):
    """
    Take the rolling mean, using a window of size n, of the given column.

    :param col:
    :param n:
    :return:
    """
    return pd.rolling_mean(col, n)


def lag(col, n):
    """
    Lag the given column by n rows.

    :param col:
    :param n:
    :return:
    """
    return col.shift(n)


def as_string(col):
    """
    ...

    :param col:
    :param n:
    :return:
    """
    return col.astype(str)


def as_int(col):
    """
    ...

    :param col:
    :param n:
    :return:
    """
    return col.astype(int)


def as_date(col):
    """
    Convert given column to a date type column.

    :param col: Pandas Series
    :return: Pandas Series
    """
    col = pd.to_datetime(col)
    return col


def day_of_week(col):
    new_col = col.dt.dayofweek
    return new_col


def add_lead_zeroes(col):
    """
    Add leading zeroes to given column (Series)

    :param col: Pandas Series
    :return: Pandas Series
    """
    return col.map("{:02}".format)


def category_bins(col, n):
    """
    Create categories from a column of numbers using n equal width bins.

    :param col: Pandas Series
    :param n: number of bins
    :return: Pandas Series
    """
    new_col = pd.cut(col, int(n))
    return pd.Series(new_col)


def replace(col, replacement_targets, replacement_values):
    """
    Replace column values...

    :param col:
    :param replacement_dict:
    :return:
    """
    return col.replace(replacement_targets, replacement_values, inplace=True)


def drop_na(df):
    return df.dropna()