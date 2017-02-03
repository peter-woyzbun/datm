import os

TEMPLATE_DIR = os.path.dirname((os.path.abspath(__file__)))

# Templates
script_docstring = open(os.path.join(TEMPLATE_DIR, 'script_docstring.txt'), 'r').read()
imports = open(os.path.join(TEMPLATE_DIR, 'imports.txt'), 'r').read()
immutable_dataset = open(os.path.join(TEMPLATE_DIR, 'immutable_dataset.txt'), 'r').read()
transformation_src = open(os.path.join(TEMPLATE_DIR, 'transformation_src.txt'), 'r').read()
