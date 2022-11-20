Usage
=====
The application consist of 

.. _installation:
Installation
------------
**WINDOWS**

To use ERC_Predictor, first run a console with admin or root permissions.
*Suggestion for python setup:* Create and run a python virtual enviroment with the following commands:

Create python virtual enviroment:
 .. code-block:: console

   >py -m venv .venv

Activate python virtual enviroment(venv):
 .. code-block:: console

   >.venv\scripts\activate

.. note::
    If the creation and validation worked there should be (.venv) at the begining of the console line.

.. _packagesAndLibraries:
Packages and libraries instalation
------------
**WINDOWS**

To install each package use pip install on the activated python virtual enviroment.

.. code-block:: console
    
    (.venv) > pip install [library name]

    * pip install pandas
    * pip install sklearn
    * pip install -U scikit-learn
    * pip install matplotlib
    * pip install xgboost
    * pip install interpret
    * pip install interpret_community

*Sphinx instalation:*
 .. code-block:: console

   > python -m pip install sphinx
   > pip install sphinx-rtd-theme

*Update sphinx doc:*
To update any changes in the documentation use the following command:
 .. code-block:: console
    > sphinx-build -b html docs/source/ docs/build/html