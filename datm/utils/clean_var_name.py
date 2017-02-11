import re


def clean_var_name(var_str):
    """ Clean variable string so that it's a valid Python name. """
    return re.sub('\W|^(?=\d)', '_', str(var_str))
