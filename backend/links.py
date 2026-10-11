"""Keep website links exact while showing them in readable Persian.

Persian URLs are stored percent-encoded (the form the site itself serves). Models
and humans tend to copy the decoded form and silently drop invisible characters
such as the zero-width non-joiner, or omit the scheme. Every URL that reaches the
user is mapped back to the exact known permalink when possible.
"""

import re
import unicodedata
from typing import Dict, Iterable, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

# Characters that differ between "looks the same" Persian spellings.
_INVISIBLE = dict.fromkeys(map(ord, "‌‍‎‏ـ "), None)
# slug separators are unreliable in copied Persian URLs ("درخواست-نمایندگی" vs "در-خواست-نمایندگی")
_SEPARATORS = dict.fromkeys(map(ord, "- _"), None)
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی"})
_TRAILING_PUNCT = ".,;:!?،؛؟»)]}'\""


def normalize_url(url: str) -> str:
    """Return a clickable, correctly percent-encoded absolute URL."""
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.IGNORECASE):
        url = "https://" + url.lstrip("/")
    parts = urlsplit(url)
    # unquote first so already-encoded input is not double-encoded
    path = quote(unquote(parts.path), safe="/-_.~!$&'()*+,;=:@")
    query = quote(unquote(parts.query), safe="=&-_.~!$'()*+,;:@/?")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path or "/", query, parts.fragment))


def display_url(url: str) -> str:
    """Readable form of a URL: no scheme, Persian characters decoded."""
    if not url:
        return ""
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.")
    text = host + unquote(parts.path)
    if parts.query:
        text += "?" + unquote(parts.query)
    return text


def _canonical(url: str) -> str:
    """Comparison key that ignores scheme, www, case, slashes and Persian spelling variants."""
    parts = urlsplit(normalize_url(url))
    text = parts.netloc.removeprefix("www.") + unquote(parts.path).rstrip("/")
    if parts.query:
        text += "?" + unquote(parts.query)
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE).translate(_ARABIC_TO_PERSIAN).translate(_SEPARATORS)
    return text.lower()


class LinkResolver:
    """Maps any spelling of a known URL to its exact permalink."""

    def __init__(self, urls: Iterable[str] = (), site_url: str = ""):
        self.site_host = urlsplit(normalize_url(site_url)).netloc.removeprefix("www.") if site_url else ""
        self._known: Dict[str, str] = {}
        for url in urls:
            self.add(url)

    def add(self, url: str) -> None:
        if url:
            url = url.strip()
            # keep the site's own spelling when it is already a valid absolute URL
            exact = url if url.isascii() and re.match(r"^https?://", url, re.IGNORECASE) else normalize_url(url)
            self._known.setdefault(_canonical(exact), exact)

    def resolve(self, url: str) -> str:
        """Exact known permalink for url, or the normalized url if it is not known."""
        if not url:
            return ""
        return self._known.get(_canonical(url), normalize_url(url))

    def is_known(self, url: str) -> bool:
        return bool(url) and _canonical(url) in self._known

    def fix_links(self, text: str) -> str:
        """Rewrite every URL in model output as a markdown link to the exact permalink."""

        def link(url: str) -> str:
            exact = self.resolve(url)
            return f"[{display_url(exact)}]({exact})"

        def replace_markdown(match: re.Match) -> str:
            label, url = match.group(1), match.group(2).strip()
            exact = self.resolve(url)
            # a label that is itself a URL is replaced with the readable form of the exact link
            if _looks_like_url(label, self.site_host):
                label = display_url(exact)
            return f"[{label}]({exact})"

        # 1. existing markdown links; protect them from step 2 with placeholders
        placeholders = []

        def stash(match: re.Match) -> str:
            placeholders.append(replace_markdown(match))
            return f"\x00{len(placeholders) - 1}\x00"

        text = re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", stash, text)

        # 2. bare URLs (with a scheme, or a bare site host)
        host = re.escape(self.site_host) if self.site_host else None
        bare = r"https?://[^\s<>\"'()\[\]]+"
        if host:
            bare = rf"(?:{bare}|(?<![\w/.@])(?:www\.)?{host}(?:/[^\s<>\"'()\[\]]*)?)"

        def replace_bare(match: re.Match) -> str:
            url = match.group(0)
            trailing = ""
            while url and url[-1] in _TRAILING_PUNCT:
                trailing = url[-1] + trailing
                url = url[:-1]
            return link(url) + trailing if url else match.group(0)

        text = re.sub(bare, replace_bare, text, flags=re.IGNORECASE)
        return re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], text)


def _looks_like_url(text: str, site_host: Optional[str]) -> bool:
    text = text.strip().lower()
    return text.startswith(("http://", "https://", "www.")) or bool(site_host and text.startswith(site_host))
