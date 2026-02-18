Contributing
============

Contributions are welcome! Here's how to get started.

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/rishi-s8/agentbuilder.git
   cd agentbuilder
   pip install -e ".[dev,docs]"

Code Style
----------

The project uses **black** for formatting and **isort** for import
sorting:

.. code-block:: bash

   black src/ tests/
   isort src/ tests/

Running Tests
-------------

.. code-block:: bash

   pytest

Tests are in the ``tests/`` directory and use pytest with coverage
reporting.

Docstring Conventions
---------------------

All docstrings follow the `Google Python Style Guide
<https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

Include:

- A one-line summary.
- ``Args:`` section for all parameters.
- ``Returns:`` section describing the return value.
- ``Raises:`` section if the function raises exceptions.
- ``Example::`` blocks for public API functions.
- ``Note:`` sections for important caveats.

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make html
   open _build/html/index.html

Pull Request Process
--------------------

1. Fork the repository and create a feature branch.
2. Make your changes with tests.
3. Run ``black``, ``isort``, and ``pytest``.
4. Submit a pull request with a clear description.

License
-------

By contributing, you agree that your contributions will be licensed under
the CC BY-SA 4.0 license.
