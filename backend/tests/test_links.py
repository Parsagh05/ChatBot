import pytest

from links import LinkResolver, display_url, normalize_url

SERVICE = "https://tabenergy.ir/%d8%ae%d8%af%d9%85%d8%a7%d8%aa/774/"  # خدمات/774
UNCATEGORIZED = "https://tabenergy.ir/%d8%af%d8%b3%d8%aa%d9%87%d8%a8%d9%86%d8%af%db%8c-%d9%86%d8%b4%d8%af%d9%87/1692/"  # دستهبندی-نشده
DEALERSHIP = "https://tabenergy.ir/%d8%af%d8%b1-%d8%ae%d9%88%d8%a7%d8%b3%d8%aa-%d9%86%d9%85%d8%a7%db%8c%d9%86%d8%af%da%af%db%8c/"  # در-خواست-نمایندگی


@pytest.fixture
def resolver():
    return LinkResolver([SERVICE, UNCATEGORIZED, DEALERSHIP, "https://tabenergy.ir/contact/"], site_url="https://tabenergy.ir")


@pytest.mark.parametrize(
    "typed, exact",
    [
        ("tabenergy.ir/خدمات/774/", SERVICE),  # no scheme
        ("https://tabenergy.ir/خدمات/774", SERVICE),  # decoded, no trailing slash
        ("https://www.tabenergy.ir/%D8%AE%D8%AF%D9%85%D8%A7%D8%AA/774/", SERVICE),  # www + uppercase hex
        ("tabenergy.ir/دسته‌بندی-نشده/1692/", UNCATEGORIZED),  # extra zero-width non-joiner
        ("tabenergy.ir/درخواست-نمایندگی/", DEALERSHIP),  # missing hyphen
        ("tabenergy.ir/contact", "https://tabenergy.ir/contact/"),
    ],
)
def test_resolve_returns_exact_site_url(resolver, typed, exact):
    assert resolver.is_known(typed)
    assert resolver.resolve(typed) == exact


def test_unknown_url_is_normalized_not_dropped(resolver):
    assert resolver.resolve("example.com/صفحه") == "https://example.com/%D8%B5%D9%81%D8%AD%D9%87"
    assert not resolver.is_known("example.com/صفحه")


def test_normalize_does_not_double_encode():
    assert normalize_url(SERVICE).lower() == SERVICE


def test_display_url_is_readable_persian():
    assert display_url(SERVICE) == "tabenergy.ir/خدمات/774/"
    assert display_url("https://www.tabenergy.ir/contact/") == "tabenergy.ir/contact/"


def test_fix_links_bare_url_keeps_trailing_punctuation(resolver):
    fixed = resolver.fix_links("به tabenergy.ir/خدمات/774/، مراجعه کنید.")
    assert fixed == f"به [tabenergy.ir/خدمات/774/]({SERVICE})، مراجعه کنید."


def test_fix_links_keeps_markdown_label_and_fixes_target(resolver):
    assert resolver.fix_links("[صفحه تماس](tabenergy.ir/contact)") == "[صفحه تماس](https://tabenergy.ir/contact/)"


def test_fix_links_replaces_url_label_with_readable_form(resolver):
    fixed = resolver.fix_links("[https://tabenergy.ir/خدمات/774](https://tabenergy.ir/خدمات/774)")
    assert fixed == f"[tabenergy.ir/خدمات/774/]({SERVICE})"


def test_fix_links_leaves_plain_text_alone(resolver):
    text = "پنل خورشیدی ۵۷۰ وات با راندمان بالا."
    assert resolver.fix_links(text) == text
