Installation
============

Prerequisites
-------------

- Python 3.9 or later
- An OpenAI API key (or a compatible provider)

Basic Install
-------------

.. code-block:: bash

   pip install agentbuilder

Or install from source:

.. code-block:: bash

   git clone https://github.com/rishi-s8/agentbuilder.git
   cd agentbuilder
   pip install -e .

Optional Extras
---------------

AgentBuilder provides optional dependency groups for additional features:

.. list-table::
   :header-rows: 1
   :widths: 15 40 30

   * - Extra
     - What it adds
     - Install command
   * - ``server``
     - FastAPI + Uvicorn for serving agents over HTTP
     - ``pip install agentbuilder[server]``
   * - ``code``
     - Docker SDK for sandboxed code execution
     - ``pip install agentbuilder[code]``
   * - ``dev``
     - pytest, black, isort for development
     - ``pip install agentbuilder[dev]``
   * - ``docs``
     - Sphinx + Furo for building documentation
     - ``pip install agentbuilder[docs]``

Install multiple extras at once:

.. code-block:: bash

   pip install agentbuilder[server,code]

Environment Setup
-----------------

Create a ``.env`` file in your project root:

.. code-block:: bash

   OPENAI_API_KEY=your_key_here
   MODEL=gpt-4o-mini
   BASE_URL=https://api.openai.com/v1

AgentBuilder uses ``python-dotenv`` to load these automatically when no
``api_key`` is passed explicitly.

Verification
------------

.. code-block:: python

   from agentbuilder.utils import create_agent
   print("AgentBuilder installed successfully!")
