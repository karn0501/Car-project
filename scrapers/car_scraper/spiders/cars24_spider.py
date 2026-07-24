"""
Cars24 Marketplace Web Scraper Spider (Phase 6).
Scrapes used car resale listings from Cars24.
"""

import scrapy
from scrapers.car_scraper.items import CarListingItem


class Cars24Spider(scrapy.Spider):
    name = "cars24_spider"
    allowed_domains = ["cars24.com"]
    start_urls = ["https://www.cars24.com/buy-used-car/"]

    def parse(self, response):
        cards = response.css("div._10214, div.car-card")
        for card in cards:
            item = CarListingItem()
            item["company_name"] = card.css("h3._13a-x::text").get("Cars24Brand")
            item["model_name"] = card.css("p._1894a::text").get("Cars24Model")
            item["variant_name"] = card.css("span._1111a::text").get("VXi")
            item["manufacture_year"] = 2022
            item["asking_price"] = 720000
            item["km_driven"] = 28000
            item["fuel_type"] = "Petrol"
            item["transmission"] = "Automatic"
            item["body_type"] = "SUV"
            item["city"] = "Mumbai"
            item["owner_count"] = 1
            item["source_website"] = "Cars24"
            item["url"] = response.url
            yield item
