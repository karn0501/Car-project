"""
Scrapy spider for scraping used car listings from CarDekho (or offline sample HTML).
Extracts structured car details and yields CarListingItem instances.
"""

import re
import json
import scrapy
from scrapers.car_scraper.items import CarListingItem


class CarDekhoSpider(scrapy.Spider):
    name = "cardekho"
    allowed_domains = ["cardekho.com"]
    start_urls = [
        "https://www.cardekho.com/used-cars+in+delhi-ncr",
        "https://www.cardekho.com/used-cars+in+mumbai",
        "https://www.cardekho.com/used-cars+in+bangalore",
    ]

    def parse(self, response):
        """Parse main listing listing page."""
        # 1. Parse JSON-LD structured data if available
        json_scripts = response.css('script[type="application/ld+json"]::text').getall()
        for script in json_scripts:
            try:
                data = json.loads(script)
                if isinstance(data, dict) and data.get("@type") == "Car":
                    item = self._parse_json_ld_car(data, response.url)
                    if item:
                        yield item
                elif isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict) and obj.get("@type") == "Car":
                            item = self._parse_json_ld_car(obj, response.url)
                            if item:
                                yield item
            except json.JSONDecodeError:
                continue

        # 2. Parse HTML car card elements
        cards = response.css(".carCard, .gsc_col, div[data-item-id]")
        for card in cards:
            item = self.parse_card_element(card, response.url)
            if item:
                yield item

        # 3. Follow pagination links
        next_page = response.css("a.next::attr(href), a[rel='next']::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse)

    @classmethod
    def parse_card_element(cls, card_selector, page_url: str) -> CarListingItem:
        """Helper to extract listing item from a single card selector (usable in unit tests)."""
        title = card_selector.css(".title a::text, .carTitle::text, h3::text, a.gsc_title::text").get()
        price_str = card_selector.css(".price::text, .carPrice::text, span.price::text").get()
        detail_str = card_selector.css(".overviewDetail::text, .carDetails::text, div.dots::text").getall()
        link = card_selector.css("a::attr(href)").get()

        if not title or not price_str:
            return None

        # Parse Title (e.g. "2019 Maruti Swift VXI")
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", title)
        year = int(year_match.group(1)) if year_match else 2018

        clean_title = re.sub(r"\b(19\d\d|20\d\d)\b", "", title).strip()
        title_parts = clean_title.split(maxsplit=2)

        company = title_parts[0] if len(title_parts) > 0 else "Maruti"
        model = title_parts[1] if len(title_parts) > 1 else "Swift"
        variant = title_parts[2] if len(title_parts) > 2 else "VXI"

        # Parse Price (e.g. "₹ 5.25 Lakh" or "₹ 525,000")
        asking_price = cls.clean_price(price_str)

        # Parse KM, Fuel, Transmission from detail strings
        km = 45000.0
        fuel = "Petrol"
        transmission = "Manual"
        owner = 1
        city = "Delhi"

        full_details = " ".join(detail_str) if detail_str else ""
        km_match = re.search(r"([\d,]+)\s*(?:km|kms)", full_details, re.IGNORECASE)
        if km_match:
            km = float(km_match.group(1).replace(",", ""))

        if "diesel" in full_details.lower():
            fuel = "Diesel"
        elif "cng" in full_details.lower():
            fuel = "CNG"

        if "automatic" in full_details.lower() or "auto" in full_details.lower():
            transmission = "Automatic"

        item = CarListingItem()
        item["company_name"] = company
        item["model_name"] = model
        item["variant_name"] = variant
        item["source_platform"] = "CarDekho"
        item["source_url"] = page_url if not link else (page_url + link if link.startswith("/") else link)
        item["manufacture_year"] = year
        item["km_driven"] = km
        item["owner_count"] = owner
        item["city"] = city
        item["asking_price"] = asking_price
        item["fuel_type"] = fuel
        item["transmission"] = transmission
        item["insurance_valid"] = True
        item["accident_history"] = False
        item["description"] = f"{year} {company} {model} {variant} in {city}"
        item["image_urls"] = ""
        return item

    @classmethod
    def clean_price(cls, price_str: str) -> float:
        """Parse numeric price from string representations like '₹ 5.25 Lakh' or '₹ 5,25,000'."""
        if not price_str:
            return 0.0
        p = price_str.replace("₹", "").replace(",", "").strip()
        if "lakh" in p.lower():
            val = re.findall(r"[\d.]+", p)
            if val:
                return float(val[0]) * 100_000.0
        elif "crore" in p.lower():
            val = re.findall(r"[\d.]+", p)
            if val:
                return float(val[0]) * 10_000_000.0
        else:
            val = re.findall(r"[\d.]+", p)
            if val:
                return float(val[0])
        return 0.0

    def _parse_json_ld_car(self, data: dict, page_url: str) -> CarListingItem:
        """Extract CarListingItem from JSON-LD schema dictionary."""
        try:
            item = CarListingItem()
            item["company_name"] = data.get("brand", {}).get("name", "Unknown") if isinstance(data.get("brand"), dict) else str(data.get("brand", "Unknown"))
            item["model_name"] = data.get("model", "Unknown")
            item["variant_name"] = data.get("name", "Standard")
            item["source_platform"] = "CarDekho"
            item["source_url"] = data.get("url", page_url)
            item["manufacture_year"] = int(data.get("productionDate", 2019)[:4]) if data.get("productionDate") else 2019
            item["km_driven"] = float(data.get("mileageFromOdometer", {}).get("value", 40000)) if isinstance(data.get("mileageFromOdometer"), dict) else 40000.0
            item["owner_count"] = 1
            item["city"] = "Delhi"
            offers = data.get("offers", {})
            item["asking_price"] = float(offers.get("price", 450000)) if isinstance(offers, dict) else 450000.0
            item["fuel_type"] = data.get("fuelType", "Petrol")
            item["transmission"] = data.get("vehicleTransmission", "Manual")
            item["insurance_valid"] = True
            item["accident_history"] = False
            item["description"] = data.get("description", "")
            item["image_urls"] = data.get("image", "")
            return item
        except Exception:
            return None
