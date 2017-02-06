###############
Transformations
###############

Introduction


Manipulation Sets
=================

Manipulation sets are similar to chained "data manipulation verb" functions from the excellent `dplyr` R package. They
are ordered sets of instructions that are applied to the parent dataset one at a time (i.e function composition).

Manipulation Types
------------------

Manipulation types correspond to common data manipulation tasks.

Filter
~~~~~~

The `filter` manipulation subsets data based on the provided conditions.

.. image:: images/manipulation_types/filter.png

.. function:: filter(conditions)

   Return only rows meeting given conditions

   :param conditions: One or more conditional :doc:`manipulation-expressions` separated by commas.

Example:

.. image:: images/manipulation_types/filter_ex_1.png


Select
~~~~~~

The `select` manipulation subsets the dataset, keeping only those columns given.

.. image:: images/manipulation_types/select.png

.. function:: select(columns)

   Keeps only the selected columns.

   :param columns: Comma separated list of column names.

Example
^^^^^^^

.. image:: images/manipulation_types/select_ex_1.png

Create
~~~~~~

The `create` manipulation allows new columns to be defined, or existing columns to be altered.

.. image:: images/manipulation_types/create.png

.. function:: create(column_name, column_definition)

   Creates a new column or alters/replaces existing column.

   :param column_name: The name of the column to create/alter.
   :param column_definition: Expression definition column...

Sort By
~~~~~~~

The `sort by` manipulation sorts the dataset based on given columns. A minus (`-`) in front of a column name indicates
that the sort on that column should be descending.

.. image:: images/manipulation_types/sort.png

.. function:: sort_by(columns)

   :param columns: A comma separated list of column names.

Example:

.. image:: images/manipulation_types/sort_ex_1.png

Rename
~~~~~~

The `rename` manipulation renames the given column.

.. image:: images/manipulation_types/rename.png

.. function:: rename(old_column_name, new_column_name)

   :param old_column_name: The name of the existing column.
   :param new_column_name: New name of existing column.

Example:

.. image:: images/manipulation_types/rename_ex_1.png

SQL Queries
===========




...

