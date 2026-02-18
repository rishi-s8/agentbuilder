"""Tests for AgentTool - local sub-agent delegation."""

import json
from unittest.mock import MagicMock

import pytest

from agentbuilder.Tools.agent_tool import AgentTool
from agentbuilder.Tools.base import Response


class TestAgentTool:
    """Tests for AgentTool class."""

    def _make_mock_agent(self, run_return="Sub-agent response"):
        """Create a mock AgenticLoop."""
        agent = MagicMock()
        agent.run.return_value = run_return
        return agent

    def test_creation(self):
        """Test AgentTool creation with a mock sub-agent."""
        agent = self._make_mock_agent()

        tool = AgentTool(
            agent=agent,
            name="test_agent",
            description="A test sub-agent",
        )

        assert tool.name == "test_agent"
        assert tool.description == "A test sub-agent"
        assert tool.parameters["type"] == "object"
        assert "task" in tool.parameters["properties"]
        assert "task" in tool.parameters["required"]

    def test_to_openai_format(self):
        """Test AgentTool produces correct OpenAI tool format."""
        agent = self._make_mock_agent()
        tool = AgentTool(agent=agent, name="helper", description="Helps with things")

        fmt = tool.to_openai_format()

        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "helper"
        assert fmt["function"]["description"] == "Helps with things"
        assert "task" in fmt["function"]["parameters"]["properties"]

    def test_delegate_calls_reset_then_run(self):
        """Test _delegate() calls agent.reset() then agent.run(task)."""
        agent = self._make_mock_agent(run_return="42")
        tool = AgentTool(agent=agent, name="math", description="Math agent")

        result = tool._delegate(task="What is 6 * 7?")

        agent.reset.assert_called_once()
        agent.run.assert_called_once_with("What is 6 * 7?")
        assert result == "42"

    def test_execute_calls_delegate(self):
        """Test execute() works through the Tool.execute() path."""
        agent = self._make_mock_agent(run_return="Done")
        tool = AgentTool(agent=agent, name="worker", description="Does work")

        response = tool.execute(task="Do the thing")

        assert isinstance(response, Response)
        assert response.success is True
        assert response.data == "Done"

    def test_execute_in_tool_map(self):
        """Test AgentTool works within a tool_map lookup + execute pattern."""
        agent = self._make_mock_agent(run_return="Result from sub-agent")
        tool = AgentTool(agent=agent, name="sub_agent", description="Sub agent")

        tool_map = {tool.name: tool}

        assert "sub_agent" in tool_map
        response = tool_map["sub_agent"].execute(task="Hello")
        assert response.success is True
        assert response.data == "Result from sub-agent"

    def test_error_propagation(self):
        """Test error propagation from sub-agent."""
        agent = self._make_mock_agent()
        agent.run.side_effect = RuntimeError("Sub-agent crashed")
        tool = AgentTool(agent=agent, name="broken", description="Broken agent")

        response = tool.execute(task="Crash please")

        assert response.success is False
        assert "Sub-agent crashed" in response.error


class TestCreateAgentTool:
    """Tests for create_agent_tool() factory."""

    def test_create_agent_tool(self, mocker):
        """Test create_agent_tool() creates an AgentTool with a properly configured agent."""
        mock_agent = MagicMock()
        mock_create_agent = mocker.patch(
            "agentbuilder.utils.create_agent", return_value=mock_agent
        )

        from agentbuilder.utils import create_agent_tool

        tool = create_agent_tool(
            name="math_expert",
            description="Solves math problems",
            model_name="gpt-4",
            tools=[],
            system_prompt="You are a math expert.",
            max_iterations=10,
        )

        assert isinstance(tool, AgentTool)
        assert tool.name == "math_expert"
        assert tool.description == "Solves math problems"
        assert tool.agent is mock_agent

        mock_create_agent.assert_called_once_with(
            model_name="gpt-4",
            tools=[],
            system_prompt="You are a math expert.",
            max_iterations=10,
        )
