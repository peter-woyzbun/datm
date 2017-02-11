.. datm documentation master file, created by
   sphinx-quickstart on Wed Jan 25 13:29:50 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

What is Datm?
=============

Datm is a graph-based, in-browser, data manipulation and visualization application. It's written in Python and uses the
django web framework. It allows the user to:

* Create "projects" and add datasets to them (see :ref:`add-datasets` demo).
* Transform datasets using "manipulation sets" (similar to dplyr, see :ref:`transform-datasets-ms` demo).
* Transform datasets using SQL (see :ref:`transform-datasets-sql` demo).
* Join two datasets (see :ref:`join-datasets` demo).
* Generate Python source code for transformed datasets (see :ref:`generate-source` demo).
* Access datasets easily from Python.
* Visualize datasets (see :ref:`dataset-visualization` demo).


See the :doc:`tutorial` for an overview of Datm's features.


.. _add-datasets:
Add Datasets
------------

.. image:: images/demo_gifs/dataset_upload.gif


.. _transform-datasets-ms:
Transform Datasets (Manipulation Set)
-------------------------------------

.. image:: images/demo_gifs/manipulation_set.gif

.. _transform-datasets-sql:
Transform Datasets (SQL)
------------------------



.. _join-datasets:
Join Datasets
-------------


.. _generate-source:
Generate Source Code
--------------------


.. _dataset-visualization:
Visualize Datasets
------------------


Contents:

.. toctree::
   :maxdepth: 2

   getting-started
   tutorial
   project-graphs
   datasets
   transformations
   manipulation-expressions
