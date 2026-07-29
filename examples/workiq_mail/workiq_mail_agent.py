"""Interactive AgentBuilder agent backed by Azure Foundry and WorkIQ Mail."""

import argparse
import json
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

from agentbuilder.Tools.mcp import MCPToolSet
from agentbuilder.utils import create_agent

DEFAULT_AZURE_FOUNDRY_ENDPOINT = (
    "https://rishisharma-test-resource.services.ai.azure.com/openai/v1"
)
DEFAULT_AZURE_FOUNDRY_DEPLOYMENT = "gpt-chat-latest"

READ_ONLY_MAIL_TOOLS = [
    "GetAttachments",
    "GetMessage",
    "SearchMessages",
    "SearchMessagesQueryParameters",
]

MUTATING_MAIL_TOOLS = [
    "AddDraftAttachments",
    "CreateDraftMessage",
    "DeleteAttachment",
    "DeleteMessage",
    "FlagEmail",
    "ForwardMessage",
    "ForwardMessageWithFullThread",
    "ReplyAllToMessage",
    "ReplyAllWithFullThread",
    "ReplyToMessage",
    "ReplyWithFullThread",
    "SendDraftMessage",
    "SendEmailWithAttachments",
    "UpdateDraft",
    "UpdateMessage",
    "UploadAttachment",
    "UploadLargeAttachment",
]


def confirm_mail_change(tool_name, arguments):
    """Ask for approval before a tool changes or sends mail."""
    preview = json.dumps(arguments, indent=2, default=str)
    if len(preview) > 2000:
        preview = f"{preview[:2000]}\n... [truncated]"
    print(f"\nApproval required for {tool_name}:\n{preview}")
    return input("Run this mail tool? [y/N] ").strip().lower() in {"y", "yes"}


def parse_args():
    default_config = Path(__file__).resolve().parents[3] / ".mcp.json"
    parser = argparse.ArgumentParser(
        description="Run an Azure Foundry agent with WorkIQ Mail MCP tools."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to the MCP config (default: {default_config})",
    )
    parser.add_argument(
        "--server",
        default="workiq_mail",
        help="MCP server name in the config (default: workiq_mail)",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Expose only search and read tools; no mail can be changed or sent",
    )
    parser.add_argument(
        "--prompt",
        help="Run a single prompt non-interactively and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    endpoint = os.getenv(
        "AZURE_FOUNDRY_ENDPOINT",
        DEFAULT_AZURE_FOUNDRY_ENDPOINT,
    )
    deployment = os.getenv(
        "AZURE_FOUNDRY_DEPLOYMENT",
        DEFAULT_AZURE_FOUNDRY_DEPLOYMENT,
    )
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential,
        "https://ai.azure.com/.default",
    )

    with MCPToolSet.from_config(
        args.config,
        args.server,
        confirmation_callback=confirm_mail_change,
        confirmation_tools=MUTATING_MAIL_TOOLS,
    ) as mail:
        tools = mail.tools
        system_prompt = (
            "You are a careful Outlook mail assistant. Use the available MCP "
            "tools to search, read, draft, reply to, forward, update, and send "
            "mail. Never claim an action succeeded unless its tool result says "
            "it succeeded. Mutating tools require terminal approval, so explain "
            "the intended action clearly before calling them."
        )
        if args.read_only:
            tools = [
                tool for tool in mail.tools if tool.name in set(READ_ONLY_MAIL_TOOLS)
            ]
            system_prompt = (
                "You are a careful read-only Outlook mail assistant. You can only "
                "search and read mail; you cannot send, reply, forward, delete, "
                "flag, or otherwise change anything. Never claim an action "
                "succeeded unless its tool result says it succeeded."
            )

        agent = create_agent(
            model_name=deployment,
            api_key=token_provider,
            base_url=endpoint,
            tools=tools,
            system_prompt=system_prompt,
            verbose=True,
            max_iterations=30,
        )

        mode = "read-only" if args.read_only else "read/write"
        print(f"Connected to {args.server} with {len(tools)} tools ({mode}).")
        print(f"Azure Foundry deployment: {deployment}")

        if args.prompt:
            print(f"\nAgent> {agent.run(args.prompt)}")
            return

        print("Enter /reset to clear the conversation or /quit to exit.")

        while True:
            try:
                message = input("\nYou> ").strip()
            except EOFError, KeyboardInterrupt:
                print()
                break

            if not message:
                continue
            if message in {"/quit", "/exit"}:
                break
            if message == "/reset":
                agent.reset()
                print("Conversation reset.")
                continue

            print(f"\nAgent> {agent.run(message)}")


if __name__ == "__main__":
    main()
