import pytest

from product_db import count_products, get_product_by_id, import_products, init_db, search_products

PRODUCTS = [
    {
        "id": 903,
        "name": "سانورتر 6 کیلو وات برند GROWATT مدل SPF 6000",
        "url": "https://tabenergy.ir/product/%d8%b3%d8%a7%d9%86%d9%88%d8%b1%d8%aa%d8%b1/",
        "categories": ["اینورتر منفصل از شبکه"],
        "short_description": "اینورتر خورشیدی همه‌کاره",
        "description": "توان خروجی : 6000 وات\nولتاژ نامی سیستم : 48 ولت\nموج خروجی تمام سینوسی",
        "price": None,
        "currency": "تومان",
        "in_stock": True,
        "image_url": "https://tabenergy.ir/wp-content/uploads/inverter.jpg",
        "attributes": {},
    },
    {
        "id": 306,
        "name": "روشنایی خورشیدی اطلس",
        "url": "https://tabenergy.ir/product/atlas/",
        "categories": ["چراغ خورشیدی"],
        "short_description": "مناسب برای روشنایی خیابان‌ها",
        "description": "",
        "price": 2500000,
        "currency": "تومان",
        "in_stock": False,
        "image_url": None,
        "attributes": {"رنگ": "مشکی"},
    },
]


@pytest.fixture(autouse=True)
def products():
    init_db()
    import_products(PRODUCTS, force=True)
    yield
    import_products([], force=True)


def test_import_counts_and_specs():
    assert count_products() == 2
    inverter = search_products("GROWATT")[0]
    assert inverter["name_fa"] == PRODUCTS[0]["name"]
    assert inverter["source_url"] == PRODUCTS[0]["url"]
    assert inverter["image_url"] == PRODUCTS[0]["image_url"]
    assert inverter["specifications"]["توان خروجی"] == "6000 وات"
    assert inverter["specifications"]["دسته‌بندی"] == "اینورتر منفصل از شبکه"


def test_search_in_persian_and_stock_price():
    lamp = search_products("اطلس")[0]
    assert lamp["price"] == 2500000
    assert lamp["in_stock"] is False
    assert lamp["specifications"]["رنگ"] == "مشکی"
    assert get_product_by_id(lamp["id"])["name_fa"] == "روشنایی خورشیدی اطلس"


def test_reimport_same_data_is_noop_and_changes_replace():
    assert import_products(PRODUCTS) == 2
    assert import_products(PRODUCTS[:1]) == 1
    assert search_products("اطلس") == []
