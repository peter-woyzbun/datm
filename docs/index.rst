.. datm documentation master file, created by
   sphinx-quickstart on Wed Jan 25 13:29:50 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

What is Datm?
=============

Datm is a graph-based, in-browser, data manipulation and visualization application. It's written in Python and uses the
django web framework. In Datm, datasets and transformations on datasets are graph nodes. A transformation has an edge
from its "parent dataset" and an edge to its "child dataset".

See the :doc:`tutorial` for an overview of Datm's features.

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src='//gifs.com/embed/pattern-base-demo-58Ek1B' frameborder='0' scrolling='no' width='1280px' height='662px' style='-webkit-backface-visibility: hidden;-webkit-transform: scale(1);' ></iframe>
    </div>



Contents:

.. toctree::
   :maxdepth: 2

   getting-started
   tutorial
   project-graphs
   transformations
   manipulation-expressions
