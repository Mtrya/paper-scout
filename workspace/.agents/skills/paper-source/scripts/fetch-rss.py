#!/usr/bin/env python3
import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

ATOM_NS = "http://www.w3.org/2005/Atom"
CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"
DC_NS = "http://purl.org/dc/elements/1.1/"


class ArticleHTMLParser(HTMLParser):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url
        self.text_parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            url = urljoin(self.base_url, href)
            if url.startswith(("http://", "https://")):
                self.links.append(url)

    def handle_data(self, data):
        self.text_parts.append(data)


def parse_html(value, base_url):
    parser = ArticleHTMLParser(base_url)
    parser.feed(value or "")
    text = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
    return text, parser.links


def parse_date(value):
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


def rss_entries(root):
    for item in root.findall("./channel/item"):
        yield {
            "url": (item.findtext("link") or item.findtext("guid") or "").strip(),
            "title": (item.findtext("title") or "").strip(),
            "author": (
                item.findtext(f"{{{DC_NS}}}creator")
                or item.findtext("author")
                or ""
            ).strip(),
            "published": (item.findtext("pubDate") or "").strip(),
            "summary_html": item.findtext("description") or "",
            "content_html": item.findtext(f"{{{CONTENT_NS}}}encoded") or "",
        }


def atom_entries(root):
    for item in root.findall(f"{{{ATOM_NS}}}entry"):
        links = item.findall(f"{{{ATOM_NS}}}link")
        link = next(
            (link for link in links if link.get("rel", "alternate") == "alternate"),
            None,
        )
        if link is None and links:
            link = links[0]
        url = (
            (link.get("href") if link is not None else None)
            or item.findtext(f"{{{ATOM_NS}}}id")
            or ""
        ).strip()
        yield {
            "url": url,
            "title": (item.findtext(f"{{{ATOM_NS}}}title") or "").strip(),
            "author": (
                item.findtext(f"{{{ATOM_NS}}}author/{{{ATOM_NS}}}name") or ""
            ).strip(),
            "published": (
                item.findtext(f"{{{ATOM_NS}}}published")
                or item.findtext(f"{{{ATOM_NS}}}updated")
                or ""
            ).strip(),
            "summary_html": item.findtext(f"{{{ATOM_NS}}}summary") or "",
            "content_html": item.findtext(f"{{{ATOM_NS}}}content") or "",
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch an RSS 2.0 or Atom feed and emit research candidates as JSON."
    )
    parser.add_argument("--feed-url", required=True, help="RSS or Atom feed URL")
    parser.add_argument("--source", required=True, help="stable source identifier")
    parser.add_argument(
        "--kind", default="blog_post", help="candidate type (default: blog_post)"
    )
    parser.add_argument(
        "--since", type=date.fromisoformat, help="keep posts on or after YYYY-MM-DD"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="maximum posts to emit (default: 10)"
    )
    parser.add_argument(
        "--canonical-host",
        help="replace article URL hosts for stable IDs across official mirrors",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main():
    args = parse_args()
    request = urllib.request.Request(
        args.feed_url, headers={"User-Agent": "paper-scout/rss"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())

    if root.tag == "rss":
        raw_entries = rss_entries(root)
    elif root.tag == f"{{{ATOM_NS}}}feed":
        raw_entries = atom_entries(root)
    else:
        raise ValueError(f"unsupported feed root element: {root.tag}")

    entries = []
    for raw in raw_entries:
        url = raw["url"]
        if not url:
            raise ValueError("feed entry has no article URL or id")
        if args.canonical_host:
            parts = urlsplit(url)
            url = urlunsplit(
                (parts.scheme, args.canonical_host, parts.path, parts.query, parts.fragment)
            )
        published_at = parse_date(raw["published"]) if raw["published"] else None
        if args.since and (not published_at or published_at.date() < args.since):
            continue

        summary, summary_links = parse_html(raw["summary_html"], url)
        content_text, content_links = parse_html(raw["content_html"], url)
        entries.append(
            {
                "id": url,
                "kind": args.kind,
                "source": args.source,
                "title": raw["title"],
                "url": url,
                "author": raw["author"],
                "published_at": published_at.isoformat() if published_at else "",
                "summary": summary or content_text[:500],
                "links": list(dict.fromkeys(summary_links + content_links)),
            }
        )
        if len(entries) >= args.limit:
            break

    print(json.dumps(entries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
