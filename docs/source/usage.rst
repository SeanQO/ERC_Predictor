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

To install all the requirements included on the requirements.txt run the conmmand

 .. code-block:: console

  (.venv)>py -m pip install -r docs/requirements.txt

.. note::
    All the project dependencies are loaded in the requirements.txt, but if any dependencie is missing use the following: 


To install each package use pip install on the activated python virtual enviroment.

.. code-block:: console
    
    (.venv) > pip install [library name]
    
    * pip install furo
    * pip install Flask
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

.. _run:
Runing
------------
**WINDOWS**
To run the project run a console on the root folder of the project, ERC_Predictor folder, and run the command:
 .. code-block:: console

   >.\src\main.py
