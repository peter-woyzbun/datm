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

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 80%; height: auto;">
        <iframe src="//www.youtube.com/embed/fxdaaFV6uyA?loop=1&playlist=fxdaaFV6uyA&autoplay=1&showinfo=0&controls=0" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>



Contents:

.. toctree::
   :maxdepth: 2

   getting-started
   tutorial
   project-graphs
   transformations
   manipulation-expressions
