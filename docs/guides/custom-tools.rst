Custom Tools
============

This guide covers both approaches to creating tools.

Using ``tool_from_function``
----------------------------

The recommended approach for most tools:

.. code-block:: python

   from pydantic import BaseModel, Field
   from agentbuilder.Tools.base import tool_from_function

   class WeatherParams(BaseModel):
       city: str = Field(description="City name")
       units: str = Field(default="celsius", description="Temperature units")

   def get_weather(params: WeatherParams) -> dict:
       """Get the current weather for a city."""
       # Your API call here
       return {"city": params.city, "temp": 22, "units": params.units}

   tool = tool_from_function(get_weather)

Rules:

- The function must accept **exactly one** parameter.
- That parameter must be a ``pydantic.BaseModel`` subclass.
- The function's docstring becomes the tool description.
- Return types are appended to the description if annotated.

Manual ``Tool`` Construction
----------------------------

For full control over the JSON schema:

.. code-block:: python

   from agentbuilder.Tools.base import Tool

   tool = Tool(
       name="search",
       description="Search the web for information",
       parameters={
           "type": "object",
           "properties": {
               "query": {
                   "type": "string",
                   "description": "Search query",
               },
               "max_results": {
                   "type": "integer",
                   "description": "Maximum results to return",
                   "default": 5,
               },
           },
           "required": ["query"],
       },
       function=my_search_function,
   )

Best Practices
--------------

1. **Clear descriptions** -- the LLM uses the description to decide when
   to call the tool. Be specific.
2. **Field descriptions** -- annotate every field with a ``description``
   in your Pydantic model.
3. **Return structured data** -- return dicts or Pydantic models so
   results are JSON-serializable.
4. **Handle errors gracefully** -- exceptions are caught by
   :meth:`~agentbuilder.Tools.base.Tool.execute` and returned as
   ``Response(success=False, error=...)``.
5. **Keep tools focused** -- one tool per action. Don't create
   "swiss-army-knife" tools.
