import os

TEMPLATE_DIR = os.path.dirname((os.path.abspath(__file__)))

# Templates
imports = open(os.path.join(TEMPLATE_DIR, 'imports.txt'), 'r').read()
immutable_dataset = open(os.path.join(TEMPLATE_DIR, 'immutable_dataset.txt'), 'r').read()
