# ruff: noqa: T201
# This script needs to print.

import random
import time
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlparse

import bs4
import requests
from html2text import HTML2Text


def get_or_raise(url: str) -> requests.Response:
    res = requests.get(url, timeout=30)

    if res.status_code != HTTPStatus.OK:
        raise ValueError("non-OK HTTP status", res.status_code, res.text)

    return res


def get_sites_from_sitemap(url: str) -> list[str]:
    res = get_or_raise(url)

    soup = bs4.BeautifulSoup(res.text, "xml")

    return [item.find("loc").text for item in soup.find_all("url")]


_sub_skip = [
    "/integrations/",
    "/get_started",
    "/guides",
    "/ecosystem",
    "/additional_resources",
]
_specific_skip = [
    # summary pages
    "https://python.langchain.com/docs/modules/",
    "https://python.langchain.com/docs/modules/data_connection/",
    "https://python.langchain.com/docs/modules/chains/how_to/",
    "https://python.langchain.com/docs/modules/chains/foundational/",
    "https://python.langchain.com/docs/modules/chains/document/",
    "https://python.langchain.com/docs/modules/chains/popular/",
    "https://python.langchain.com/docs/modules/chains/additional/",
    "https://python.langchain.com/docs/modules/agents/agent_types/",
    "https://python.langchain.com/docs/modules/agents/toolkits/",
    "https://python.langchain.com/docs/modules/callbacks/custom_chain",
    "https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/",
    "https://python.langchain.com/docs/modules/model_io/",
    "https://python.langchain.com/docs/modules/model_io/prompts/",
    "https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/format_output",
    "https://python.langchain.com/docs/use_cases",
    "https://python.langchain.com/docs/use_cases/agent_simulations/",
    "https://python.langchain.com/docs/use_cases/agents/",
    "https://python.langchain.com/docs/use_cases/apis",
    "https://python.langchain.com/docs/use_cases/autonomous_agents/",
    "https://python.langchain.com/docs/use_cases/chatbots",
    "https://python.langchain.com/docs/use_cases/code/",
    "https://python.langchain.com/docs/use_cases/more/code_writing/",
    "https://python.langchain.com/docs/use_cases/more/extraction",
    "https://python.langchain.com/docs/use_cases/more/graph",
    "https://python.langchain.com/docs/use_cases/more/self_check/",
    "https://python.langchain.com/docs/use_cases/summarization",
    "https://python.langchain.com/docs/use_cases/tabular",
    # duplicates
    "https://python.langchain.com/docs/use_cases/agent_simulations/camel_role_playing",
    "https://python.langchain.com/docs/use_cases/autonomous_agents/baby_agi",
    "https://python.langchain.com/docs/use_cases/autonomous_agents/baby_agi_with_agent",
]


def should_skip_link(url: str) -> bool:
    for sub in _sub_skip:
        if sub in url:
            return True

    if url in _specific_skip:
        return True

    return False


def should_skip_page(body: bs4.Tag | bs4.NavigableString | None) -> bool:
    if body is None:
        # This page does not have the right div.
        return True

    if (
        len(body.contents) == 2  # noqa: PLR2004  # only used once
        and body.contents[0].name == "h1"
        and body.contents[1].name == "section"
        and body.contents[1].get("class") == ["row"]
    ):
        # This page is just a navigation page.
        return True

    return False


# for postprocessing
_h2t = HTML2Text(
    baseurl="https://python.langchain.com",
    bodywidth=0,  # no wrap
)
_h2t.ignore_images = True
_h2t.mark_code = True
_h2t.skip_internal_links = True
_h2t.unicode_snob = True
code_start = "[code]"
code_end = "[/code]"


def format_soup(body: bs4.Tag | bs4.NavigableString, h2t=_h2t) -> str:
    # convert to plaintext
    text = h2t.handle("\n".join(str(thing) for thing in body.contents))

    # make code blocks use markdown backticks instead of [code][/code]
    inside_code_block = False
    skip_next_if_blank = False  # [code][/code] blocks add extra newlines before and after
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()

        if inside_code_block:
            if skip_next_if_blank:
                # We just entered a code block and we don't want the extra newline.
                skip_next_if_blank = False
                if stripped == "":
                    continue

            if stripped == code_end:
                inside_code_block = False
                if lines[-1].strip() == "":
                    # remove extra newline at end of code block
                    lines.pop()
                lines.append("```")
                continue

            # remove the extra indentation
            if line.startswith("    "):
                lines.append(line[4:])
                continue
            lines.append(line)

        if stripped == code_start:
            # starting new code block
            inside_code_block = True
            skip_next_if_blank = True
            lines.append("```")
            continue

        lines.append(line)

    return "\n".join(line.rstrip() for line in lines)


# pages manually marked as procedures rather than concept docs
manually_procedures = [
    "docs/modules/agents",
    "docs/modules/agents/tools/custom_tools",
    "docs/modules/agents/tools/human_approval",
    "docs/modules/agents/tools/multi_input_tool",
    "docs/modules/agents/tools/tools_as_openai_functions",
    "docs/modules/agents/tools/tool_input_validation",
    "docs/modules/callbacks/async_callbacks",
    "docs/modules/callbacks/custom_callbacks",
    "docs/modules/callbacks/custom_chain",
    "docs/modules/callbacks/filecallbackhandler",
    "docs/modules/callbacks/multiple_callbacks",
    "docs/modules/callbacks/tags",
    "docs/modules/callbacks/token_counting",
    "docs/modules/chains/foundational/llm_chain",
    "docs/modules/data_connection/document_loaders/csv",
    "docs/modules/data_connection/document_loaders/file_directory",
    "docs/modules/data_connection/document_loaders/html",
    "docs/modules/data_connection/document_loaders/json",
    "docs/modules/data_connection/document_loaders/markdown",
    "docs/modules/data_connection/document_loaders/pdf",
    "docs/modules/data_connection/retrievers/contextual_compression",
    "docs/modules/data_connection/retrievers/MultiQueryRetriever",
    "docs/modules/data_connection/retrievers/self_query",
    "docs/modules/data_connection/retrievers/time_weighted_vectorstore",
    "docs/modules/data_connection/retrievers/vectorstore",
    "docs/modules/data_connection/retrievers/self_query/chroma_self_query",
    "docs/modules/data_connection/retrievers/self_query/myscale_self_query",
    "docs/modules/data_connection/retrievers/self_query/pinecone",
    "docs/modules/data_connection/retrievers/self_query/qdrant_self_query",
    "docs/modules/data_connection/retrievers/self_query/weaviate_self_query",
    "docs/modules/memory",
    "docs/modules/memory/adding_memory",
    "docs/modules/memory/adding_memory_chain_multiple_inputs",
    "docs/modules/memory/agent_with_memory",
    "docs/modules/memory/agent_with_memory_in_db",
    "docs/modules/memory/types/buffer",
    "docs/modules/memory/types/buffer_window",
    "docs/modules/memory/conversational_customization",
    "docs/modules/memory/custom_memory",
    "docs/modules/memory/types/entity_summary_memory",
    "docs/modules/memory/types/kg",
    "docs/modules/memory/multiple_memory",
    "docs/modules/memory/types/summary",
    "docs/modules/memory/types/summary_buffer",
    "docs/modules/memory/types/token_buffer",
    "docs/modules/memory/types/vectorstore_retriever_memory",
    "docs/modules/model_io/models/chat/chat_model_caching",
    "docs/modules/model_io/models/chat/human_input_chat_model",
    "docs/modules/model_io/models/chat/prompts",
    "docs/modules/model_io/models/chat/streaming",
    "docs/modules/model_io/models/llms/async_llm",
    "docs/modules/model_io/models/llms/custom_llm",
    "docs/modules/model_io/models/llms/fake_llm",
    "docs/modules/model_io/models/llms/human_input_llm",
    "docs/modules/model_io/models/llms/llm_caching",
    "docs/modules/model_io/models/llms/llm_serialization",
    "docs/modules/model_io/models/llms/streaming_llm",
    "docs/modules/model_io/models/llms/token_usage_tracking",
    "docs/use_cases/agent_simulations",
    "docs/use_cases/apis",
    "docs/use_cases/autonomous_agents",
    "docs/use_cases/chatbots",
    "docs/use_cases/extraction",
    "docs/use_cases/question_answering",
    "docs/use_cases/summarization",
    "docs/use_cases/tabular",
]


def is_procedure(url: str) -> bool:
    if "how_to/" in url:
        return True

    if "use_cases/" in url:
        after = url.split("use_cases/", 2)[1]
        parts = after.split("/", 2)

        if len(parts) == 2 and parts[1] != "":  # noqa: PLR2004
            return True

    url = url.rstrip("/")

    return any(url.endswith(path) for path in manually_procedures)


def scrape_langchain():
    sitemap_url = "https://python.langchain.com/sitemap.xml"

    procedure_dir = Path("docs/procedures/full")
    concept_dir = Path("docs/concepts")

    for site_url in get_sites_from_sitemap(sitemap_url):
        if should_skip_link(site_url):
            continue

        print("scraping", site_url)
        res = get_or_raise(site_url)

        soup = bs4.BeautifulSoup(res.text, "html.parser")
        body = soup.find("div", class_="theme-doc-markdown markdown")

        if should_skip_page(body):
            continue

        doc = format_soup(body)

        path = urlparse(site_url).path[1:]  # remove initial /
        out_path = procedure_dir / path if is_procedure(site_url) else concept_dir / path
        out_path = out_path.with_name(out_path.name + ".md")
        print("saving as", out_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            f.write(doc)

        time.sleep(1.5 + 2 * random.random())


if __name__ == "__main__":
    scrape_langchain()
