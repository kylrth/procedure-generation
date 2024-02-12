# ruff: noqa: T201
# This script needs to print.

import re
import subprocess
from collections.abc import Generator
from os import PathLike
from pathlib import Path

from bs4 import BeautifulSoup
from html2text import HTML2Text
from scrape import format_soup
from tqdm import tqdm


def get_api_reference(target_dir: str | PathLike) -> Path:
    """Build the langchain docs and return the path to the resulting HTML source tree."""
    target_dir = Path(target_dir)

    if not target_dir.exists():
        # clone repo
        subprocess.check_call(
            [
                "git",
                "clone",
                "https://github.com/hwchase17/langchain",
                subprocess.list2cmdline([target_dir]),  # escape
            ]
        )
    else:
        # fetch any new tags
        subprocess.check_call(["git", "fetch"], cwd=target_dir)

    # check out fixed version
    subprocess.check_call(["git", "checkout", "v0.0.249"], cwd=target_dir)

    # build the API docs
    build_dir = target_dir / "docs" / "api_reference" / "_build"
    subprocess.check_call(["make", "html"], cwd=build_dir.parent)

    return build_dir / "html"


def get_api_paths(page: Path) -> Generator[tuple[str, Path], None, None]:
    """Read the paths to all the API documents as listed on the main page.

    Yields each page's path in the logical tree as well as the path to the file on disk.
    """
    with page.open() as f:
        content = f.read()
    soup = BeautifulSoup(content, "lxml")

    for item in soup.find_all("code", class_="py-obj"):
        # get the link and trim location anchor
        subpage = item.parent["href"].split("#", 2)[0]

        yield subpage, page.parent / subpage


_h2t = HTML2Text(
    bodywidth=0,  # no wrap
)
_h2t.ignore_emphasis = True  # Lots of italics make it hard to read the call signatures.
_h2t.ignore_images = True
_h2t.mark_code = True
_h2t.skip_internal_links = True
_h2t.unicode_snob = True

_param_re = re.compile(r"^(param \w+:.*)¶$", flags=re.MULTILINE)
_python_func_re = re.compile(r"^([^#\-\n].*)¶$", flags=re.MULTILINE)
_newline_re = re.compile(r"(\s*\n){3,}")
_trailing_whitespace_re = re.compile(r"[ \t]+$", flags=re.MULTILINE)


def get_doc(file: str | PathLike) -> str:
    with Path(file).open() as f:
        content = f.read()
    soup = BeautifulSoup(content, "lxml")

    formatted = format_soup(soup.find("section"), h2t=_h2t)

    # regex replacement formatting
    # We're going to add some markings that increase readability for humans (and maybe for LLMs?).
    # There is a ¶ at the end of the headings, which we'll use to find the headings and improve
    # formatting.

    # The doc title header doesn't need any changes.
    formatted = formatted.replace("¶\n", "\n", 1)

    # add bullet points before all parameter descriptions
    formatted = _param_re.sub(r"- \1", formatted)

    # replace all the set config options first to avoid marking them like python methods
    if "\nmodel Config" in formatted:
        begin, config_text = formatted.split("model Config", 2)
        config_text = _python_func_re.sub(r"- \1", config_text)
        formatted = begin + "model Config" + config_text

    # mark all Python methods with ##
    formatted = _python_func_re.sub(r"## \1", formatted)

    # Two blank lines between paragraphs is unnecessary.
    formatted = _newline_re.sub(r"\n\n", formatted)

    # Trailing whitespace is a waste of tokens.
    formatted = _trailing_whitespace_re.sub("", formatted)

    return formatted


if __name__ == "__main__":
    doc_root = get_api_reference("_lc_api_ref_gen")
    out_root = Path("./docs") / "api"

    files = list(get_api_paths(doc_root / "api_reference.html"))
    for name, file in tqdm(files, desc="formatting files"):
        doc = get_doc(file)

        out_path = out_root / name
        out_path = out_path.with_name(out_path.name + ".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            f.write(doc)
