Conversation Management
=======================

AgentBuilder stores conversation history as a list of
:class:`~agentbuilder.Action.base.Action` objects inside the
:class:`~agentbuilder.Client.base.BaseConversationWrapper`.

Accessing History
-----------------

.. code-block:: python

   history = agent.conversation.get_history()
   for action in history:
       print(type(action).__name__, getattr(action, 'content', '')[:50])

Get the last message:

.. code-block:: python

   last = agent.conversation.get_last_message()

Converting to OpenAI Format
----------------------------

.. code-block:: python

   messages = agent.conversation.to_messages()
   # [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]

Saving and Loading
------------------

.. code-block:: python

   # Save
   agent.conversation.save_conversation("chat.json")

   # Load (replaces current history)
   agent.conversation.load_conversation("chat.json")

Resetting
---------

Reset clears conversation history and planner state:

.. code-block:: python

   agent.reset()

This calls both ``conversation.reset()`` and ``planner.reset()``.

Adding Messages Manually
-------------------------

.. code-block:: python

   agent.conversation.add_system_message("New system context")
   agent.conversation.add_user_message("User says this")
   agent.conversation.add_assistant_message("Assistant responds")

Simple Chat (No Tools)
----------------------

Use :meth:`~agentbuilder.Client.openai_client.ConversationWrapper.send_message`
for a simple LLM call without tool orchestration:

.. code-block:: python

   from agentbuilder.Client.openai_client import ConversationWrapper

   conv = ConversationWrapper(
       model="gpt-4o-mini",
       system_prompt="You are a poet.",
   )
   reply = conv.send_message("Write a haiku about coding.")
   print(reply)
