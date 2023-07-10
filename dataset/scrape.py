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
    "https://python.langchain.com/docs/modules/model_io/",
    "https://python.langchain.com/docs/modules/model_io/prompts/",
    "https://python.langchain.com/docs/modules/data_connection/",
    "https://python.langchain.com/docs/modules/chains/how_to/",
    "https://python.langchain.com/docs/modules/chains/foundational/",
    "https://python.langchain.com/docs/modules/chains/document/",
    "https://python.langchain.com/docs/modules/chains/popular/",
    "https://python.langchain.com/docs/modules/chains/additional/",
    "https://python.langchain.com/docs/modules/agents/agent_types/",
    "https://python.langchain.com/docs/modules/agents/toolkits/",
    "https://python.langchain.com/docs/use_cases",
    "https://python.langchain.com/docs/use_cases/code/",
    # too long
    "https://python.langchain.com/docs/use_cases/agent_simulations/multiagent_authoritarian",
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
h2t = HTML2Text(
    baseurl="https://python.langchain.com",
    bodywidth=0,  # no wrap
)
h2t.ignore_images = True
h2t.mark_code = True
h2t.skip_internal_links = True
h2t.unicode_snob = True
code_start = "[code]"
code_end = "[/code]"


def format_soup(body: bs4.Tag | bs4.NavigableString | None) -> str:
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

    return "\n".join(lines)


def is_procedure(url: str) -> bool:
    if "how_to/" in url:
        return True

    if "use_cases/" in url:
        after = url.split("use_cases/", 2)[1]
        parts = after.split("/", 2)

        if len(parts) == 2 and parts[1] != "":  # noqa: PLR2004
            return True

    return False


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

        time.sleep(1.5 + 2 * random.random())  # noqa: S311  # not doing cryptography


if __name__ == "__main__":
    scrape_langchain()
