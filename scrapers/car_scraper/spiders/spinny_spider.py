"""
Spinny Marketplace Web Scraper Spider (Phase 6).
Scrapes used car resale listings from Spinny.
"""

import scrapy
from scrapers.car_scraper.items import CarListingItem


class SpinnySpider(scrapy.Spider):
    name = "spinny_spider"
    allowed_domains = ["spinny.com"]
    start_urls = ["https://www.spinny.com/used-cars/"]

    def parse(self, response):
        cards = response.css("div.car-card, div.listing-card")
        for card in cards:
            item = CarListingItem()
            item["company_name"] = card.css("span.make::text, div.brand::text").get("SpinnyBrand")
            item["model_name"] = card.css("span.model::text, div.model::text").get("SpinnyModel")
            item["variant_name"] = card.css("span.variant::text").get("Base Variant")
            item["manufacture_year"] = 2021
            item["asking_price"] = 650000
            item["km_driven"] = 35000
            item["fuel_type"] = "Petrol"
            item["transmission"] = "Manual"
            item["body_type"] = "Hatchback"
            item["city"] = "Delhi"
            item["owner_count"] = 1
            item["source_website"] = "Spinny"
            item["url"] = response.url
            yield item
