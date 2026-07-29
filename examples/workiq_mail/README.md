# Azure Foundry WorkIQ Mail Agent

This example uses:

- Azure Foundry deployment `gpt-chat-latest`
- `DefaultAzureCredential` and your existing `az login` session
- The repository-level `.mcp.json`
- The WorkIQ Mail OAuth registration and token already created by GitHub
  Copilot CLI

No model API key is required.

## Prerequisites

- Python 3.10 or later
- Azure CLI authenticated with `az login`
- Access to the Azure Foundry resource
- The `workiq_mail` server authenticated and working in GitHub Copilot CLI

## Install

From the `agentbuilder` directory:

```bash
python3 -m pip install -e ".[mcp,foundry]"
```

## Run

```bash
cd examples/workiq_mail
python3 workiq_mail_agent.py
```

The defaults are:

```text
Endpoint:   https://rishisharma-test-resource.services.ai.azure.com/openai/v1
Deployment: gpt-chat-latest
```

You can override them with `AZURE_FOUNDRY_ENDPOINT` and
`AZURE_FOUNDRY_DEPLOYMENT`.

The default MCP config path is the `.mcp.json` beside the `agentbuilder`
folder. Override it when needed:

```bash
python3 workiq_mail_agent.py --config /path/to/.mcp.json --server workiq_mail
```

To expose only search and read tools, so the agent cannot change any mail, use
`--read-only`. Combine it with `--prompt` for a single non-interactive run:

```bash
python3 workiq_mail_agent.py --read-only --prompt "What is the last email I received?"
```

Read-only tools run immediately. Sending, replying, forwarding, deleting, or
otherwise changing mail requires confirmation in the terminal. MCP OAuth
tokens remain in `~/.copilot/mcp-oauth-config`; they are not copied into this
project.

If mail authentication has fully expired, run `/mcp` in GitHub Copilot CLI and
authenticate `workiq_mail` again.
