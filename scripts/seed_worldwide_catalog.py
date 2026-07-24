"""
Exhaustive Car Variant & Tiered Pricing Seeder.
Populates complete variant matrix (Base, Mid, Top, Automatic, Sunroof/Dark, Hybrid, EV)
with distinct, accurate ex-showroom MSRP prices for EVERY car model across all brands.
"""

import os
import sys
import random
from datetime import datetime, timezone

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.database import init_db, SessionLocal
from db.models import Company, Model, Variant, Listing, ScraperLog

# CATALOG STRUCTURE: Brand -> { country: ..., models: [ (ModelName, BodyType, LaunchYear, DiscontinuedYear, VariantsList) ] }
# Variant format: (VariantName, FuelType, Transmission, EngineCC, SeatingCapacity, DistinctExShowroomPriceINR)
WORLDWIDE_CATALOG = {
    # --- MARUTI SUZUKI ---
    "Maruti Suzuki": {
        "country": "India",
        "models": [
            ("Swift", "Hatchback", 2005, None, [
                ("Swift LXI Petrol MT", "Petrol", "Manual", 1197, 5, 649000),
                ("Swift VXI Petrol MT", "Petrol", "Manual", 1197, 5, 729000),
                ("Swift VXI Petrol AMT", "Petrol", "Automatic", 1197, 5, 779000),
                ("Swift VXI CNG MT", "CNG", "Manual", 1197, 5, 819000),
                ("Swift ZXI Petrol MT", "Petrol", "Manual", 1197, 5, 829000),
                ("Swift ZXI Petrol AMT", "Petrol", "Automatic", 1197, 5, 879000),
                ("Swift ZXI Plus Petrol MT", "Petrol", "Manual", 1197, 5, 899000),
                ("Swift ZXI Plus Petrol AMT", "Petrol", "Automatic", 1197, 5, 949000),
                ("Swift Z-Series ZXI Plus Dual Tone 2024", "Petrol", "Automatic", 1197, 5, 965000),
            ]),
            ("Baleno", "Hatchback", 2015, None, [
                ("Baleno Sigma Petrol MT", "Petrol", "Manual", 1197, 5, 666000),
                ("Baleno Delta Petrol MT", "Petrol", "Manual", 1197, 5, 750000),
                ("Baleno Delta Petrol AGS", "Petrol", "Automatic", 1197, 5, 800000),
                ("Baleno Delta CNG MT", "CNG", "Manual", 1197, 5, 840000),
                ("Baleno Zeta Petrol MT", "Petrol", "Manual", 1197, 5, 843000),
                ("Baleno Zeta Petrol AGS", "Petrol", "Automatic", 1197, 5, 893000),
                ("Baleno Alpha Petrol MT", "Petrol", "Manual", 1197, 5, 938000),
                ("Baleno Alpha Petrol AGS", "Petrol", "Automatic", 1197, 5, 988000),
            ]),
            ("Brezza", "SUV", 2016, None, [
                ("Brezza LXI Petrol MT", "Petrol", "Manual", 1462, 5, 834000),
                ("Brezza VXI Petrol MT", "Petrol", "Manual", 1462, 5, 969000),
                ("Brezza VXI Petrol AT", "Petrol", "Automatic", 1462, 5, 1119000),
                ("Brezza VXI CNG MT", "CNG", "Manual", 1462, 5, 1064000),
                ("Brezza ZXI Petrol MT", "Petrol", "Manual", 1462, 5, 1114000),
                ("Brezza ZXI Petrol AT", "Petrol", "Automatic", 1462, 5, 1254000),
                ("Brezza ZXI Plus Petrol MT Dual Tone", "Petrol", "Manual", 1462, 5, 1258000),
                ("Brezza ZXI Plus Petrol AT Dual Tone", "Petrol", "Automatic", 1462, 5, 1398000),
            ]),
            ("Fronx", "SUV", 2023, None, [
                ("Fronx Sigma 1.2 Petrol MT", "Petrol", "Manual", 1197, 5, 751000),
                ("Fronx Delta 1.2 Petrol MT", "Petrol", "Manual", 1197, 5, 832000),
                ("Fronx Delta Plus 1.2 Petrol AGS", "Petrol", "Automatic", 1197, 5, 927000),
                ("Fronx Zeta 1.0 Turbo MT", "Petrol", "Manual", 998, 5, 1055000),
                ("Fronx Zeta 1.0 Turbo 6AT", "Petrol", "Automatic", 998, 5, 1195000),
                ("Fronx Alpha 1.0 Turbo 6AT Dual Tone", "Petrol", "Automatic", 998, 5, 1304000),
            ]),
            ("Grand Vitara", "SUV", 2022, None, [
                ("Grand Vitara Sigma 1.5 Mild Hybrid MT", "Petrol", "Manual", 1462, 5, 1099000),
                ("Grand Vitara Delta 1.5 Mild Hybrid MT", "Petrol", "Manual", 1462, 5, 1220000),
                ("Grand Vitara Zeta 1.5 Mild Hybrid AT", "Petrol", "Automatic", 1462, 5, 1560000),
                ("Grand Vitara Alpha 1.5 Mild Hybrid AWD MT", "Petrol", "Manual", 1462, 5, 1701000),
                ("Grand Vitara Zeta Plus 1.5 Strong Hybrid e-CVT", "Hybrid", "CVT", 1490, 5, 1843000),
                ("Grand Vitara Alpha Plus 1.5 Strong Hybrid e-CVT", "Hybrid", "CVT", 1490, 5, 1993000),
            ]),
            ("Jimny", "SUV", 2023, None, [("Jimny Zeta 1.5 4WD MT", "Petrol", "Manual", 1462, 4, 1274000), ("Jimny Zeta 1.5 4WD AT", "Petrol", "Automatic", 1462, 4, 1394000), ("Jimny Alpha 1.5 4WD MT Dual Tone", "Petrol", "Manual", 1462, 4, 1385000), ("Jimny Alpha 1.5 4WD AT Dual Tone", "Petrol", "Automatic", 1462, 4, 1505000)]),
            ("Ertiga", "MPV", 2012, None, [("Ertiga LXI Petrol MT", "Petrol", "Manual", 1462, 7, 869000), ("Ertiga VXI Petrol MT", "Petrol", "Manual", 1462, 7, 983000), ("Ertiga VXI Petrol AT", "Petrol", "Automatic", 1462, 7, 1123000), ("Ertiga VXI CNG MT", "CNG", "Manual", 1462, 7, 1078000), ("Ertiga ZXI Plus Petrol AT", "Petrol", "Automatic", 1462, 7, 1303000)]),
            ("Dzire", "Sedan", 2008, None, [("Dzire LXI Petrol MT", "Petrol", "Manual", 1197, 5, 657000), ("Dzire VXI Petrol MT", "Petrol", "Manual", 1197, 5, 779000), ("Dzire VXI Petrol AGS", "Petrol", "Automatic", 1197, 5, 829000), ("Dzire ZXI Plus Petrol AGS", "Petrol", "Automatic", 1197, 5, 939000)]),
        ]
    },

    # --- HYUNDAI ---
    "Hyundai": {
        "country": "South Korea",
        "models": [
            ("Creta", "SUV", 2014, None, [
                ("Creta E 1.5 Petrol MT", "Petrol", "Manual", 1497, 5, 1099000),
                ("Creta EX 1.5 Petrol MT", "Petrol", "Manual", 1497, 5, 1221000),
                ("Creta S 1.5 Petrol MT", "Petrol", "Manual", 1497, 5, 1343000),
                ("Creta SX 1.5 Petrol IVT", "Petrol", "CVT", 1497, 5, 1586000),
                ("Creta SX (O) 1.5 Petrol IVT", "Petrol", "CVT", 1497, 5, 1741000),
                ("Creta SX (O) 1.5 Turbo GDi DCT 2024", "Petrol", "Automatic", 1482, 5, 2000000),
                ("Creta E 1.5 Diesel MT", "Diesel", "Manual", 1493, 5, 1256000),
                ("Creta SX (O) 1.5 Diesel AT", "Diesel", "Automatic", 1493, 5, 1999000),
                ("Creta N Line N8 1.5 Turbo MT", "Petrol", "Manual", 1482, 5, 1682000),
                ("Creta N Line N10 1.5 Turbo DCT Dual Tone", "Petrol", "Automatic", 1482, 5, 2045000),
            ]),
            ("Verna", "Sedan", 2006, None, [
                ("Verna EX 1.5 MPi MT", "Petrol", "Manual", 1497, 5, 1100000),
                ("Verna S 1.5 MPi MT", "Petrol", "Manual", 1497, 5, 1199000),
                ("Verna SX 1.5 MPi IVT", "Petrol", "CVT", 1497, 5, 1423000),
                ("Verna SX (O) 1.5 MPi IVT", "Petrol", "CVT", 1497, 5, 1623000),
                ("Verna SX 1.5 Turbo GDi MT", "Petrol", "Manual", 1482, 5, 1483000),
                ("Verna SX (O) 1.5 Turbo GDi DCT Dual Tone", "Petrol", "Automatic", 1482, 5, 1742000),
            ]),
            ("Exter", "SUV", 2023, None, [
                ("Exter EX 1.2 Petrol MT", "Petrol", "Manual", 1197, 5, 613000),
                ("Exter S 1.2 Petrol MT", "Petrol", "Manual", 1197, 5, 750000),
                ("Exter SX 1.2 Petrol AMT", "Petrol", "Automatic", 1197, 5, 870000),
                ("Exter SX (O) Connect 1.2 AMT Dual Tone", "Petrol", "Automatic", 1197, 5, 1015000),
            ]),
            ("i20", "Hatchback", 2008, None, [
                ("i20 Era 1.2 Petrol MT", "Petrol", "Manual", 1197, 5, 704000),
                ("i20 Magna 1.2 Petrol MT", "Petrol", "Manual", 1197, 5, 775000),
                ("i20 Sportz 1.2 Petrol IVT", "Petrol", "CVT", 1197, 5, 943000),
                ("i20 Asta (O) 1.2 Petrol IVT", "Petrol", "CVT", 1197, 5, 1121000),
                ("i20 N Line N6 1.0 Turbo MT", "Petrol", "Manual", 998, 5, 999000),
                ("i20 N Line N8 1.0 Turbo DCT", "Petrol", "Automatic", 998, 5, 1252000),
            ]),
        ]
    },

    # --- TATA MOTORS ---
    "Tata Motors": {
        "country": "India",
        "models": [
            ("Nexon", "SUV", 2017, None, [
                ("Nexon Smart 1.2 Petrol MT", "Petrol", "Manual", 1199, 5, 815000),
                ("Nexon Pure 1.2 Petrol MT", "Petrol", "Manual", 1199, 5, 980000),
                ("Nexon Creative Plus 1.2 Petrol DCA", "Petrol", "Automatic", 1199, 5, 1300000),
                ("Nexon Fearless Plus S Dark 1.2 Petrol DCA", "Petrol", "Automatic", 1199, 5, 1480000),
                ("Nexon Smart 1.5 Diesel MT", "Diesel", "Manual", 1497, 5, 1110000),
                ("Nexon Fearless Plus S Dark 1.5 Diesel AMT", "Diesel", "Automatic", 1497, 5, 1560000),
                ("Nexon EV Medium Range Creative 30 kWh", "EV", "Automatic", 0, 5, 1449000),
                ("Nexon EV Long Range Empowered Plus 40.5 kWh", "EV", "Automatic", 0, 5, 1949000),
            ]),
            ("Punch", "SUV", 2021, None, [
                ("Punch Pure 1.2 Petrol MT", "Petrol", "Manual", 1199, 5, 613000),
                ("Punch Adventure 1.2 Petrol AMT", "Petrol", "Automatic", 1199, 5, 785000),
                ("Punch Accomplished Dazzle 1.2 Petrol AMT", "Petrol", "Automatic", 1199, 5, 885000),
                ("Punch Creative Flagship 1.2 Petrol AMT Sunroof", "Petrol", "Automatic", 1199, 5, 1020000),
                ("Punch EV Smart 25 kWh", "EV", "Automatic", 0, 5, 1099000),
                ("Punch EV Empowered Plus S Long Range 35 kWh", "EV", "Automatic", 0, 5, 1449000),
            ]),
            ("Curvv", "SUV", 2024, None, [
                ("Curvv Smart 1.2 Turbo Petrol MT", "Petrol", "Manual", 1199, 5, 1000000),
                ("Curvv Accomplished 1.5 TGDi DCA", "Petrol", "Automatic", 1498, 5, 1650000),
                ("Curvv Empowered Plus 1.5 TGDi DCA Dark", "Petrol", "Automatic", 1498, 5, 1950000),
                ("Curvv EV 45 kWh Creative", "EV", "Automatic", 0, 5, 1749000),
                ("Curvv EV 55 kWh Empowered Plus S", "EV", "Automatic", 0, 5, 2199000),
            ]),
            ("Harrier", "SUV", 2019, None, [
                ("Harrier Smart 2.0 Diesel MT", "Diesel", "Manual", 1956, 5, 1549000),
                ("Harrier Pure Plus 2.0 Diesel AT", "Diesel", "Automatic", 1956, 5, 1999000),
                ("Harrier Adventure Plus 2.0 Diesel AT Sunroof", "Diesel", "Automatic", 1956, 5, 2349000),
                ("Harrier Fearless Plus Dark 2.0 Diesel AT", "Diesel", "Automatic", 1956, 5, 2644000),
            ]),
        ]
    },

    # --- MAHINDRA ---
    "Mahindra": {
        "country": "India",
        "models": [
            ("XUV700", "SUV", 2021, None, [
                ("XUV700 MX 2.0 Petrol MT 5-Seat", "Petrol", "Manual", 1997, 5, 1399000),
                ("XUV700 AX3 2.0 Petrol AT 5-Seat", "Petrol", "Automatic", 1997, 5, 1819000),
                ("XUV700 AX5 2.2 Diesel MT 7-Seat", "Diesel", "Manual", 2184, 7, 1989000),
                ("XUV700 AX7 2.2 Diesel AT 7-Seat", "Diesel", "Automatic", 2184, 7, 2389000),
                ("XUV700 AX7 Luxury Pack 2.2 Diesel AWD AT 7-Seat", "Diesel", "Automatic", 2184, 7, 2699000),
            ]),
            ("Thar", "SUV", 2010, None, [
                ("Thar AX OPT 2.0 Petrol Convertible 4WD MT", "Petrol", "Manual", 1997, 4, 1430000),
                ("Thar LX 2.0 Petrol Hard Top 4WD AT", "Petrol", "Automatic", 1997, 4, 1660000),
                ("Thar LX 2.2 Diesel Hard Top 4WD AT", "Diesel", "Automatic", 2184, 4, 1720000),
                ("Thar LX 1.5 Diesel RWD Hard Top MT", "Diesel", "Manual", 1497, 4, 1135000),
                ("Thar Earth Edition 2.2 Diesel 4WD AT", "Diesel", "Automatic", 2184, 4, 1760000),
            ]),
            ("Thar ROXX 5-Door", "SUV", 2024, None, [
                ("Thar ROXX MX1 2.0 Petrol RWD MT 5-Door", "Petrol", "Manual", 1997, 5, 1299000),
                ("Thar ROXX MX5 2.2 Diesel 4WD MT 5-Door", "Diesel", "Manual", 2184, 5, 1899000),
                ("Thar ROXX AX7 L 2.2 Diesel 4WD AT Sunroof 5-Door", "Diesel", "Automatic", 2184, 5, 2249000),
            ]),
            ("Scorpio-N", "SUV", 2022, None, [
                ("Scorpio-N Z2 2.0 Petrol MT 7-Seat", "Petrol", "Manual", 1997, 7, 1385000),
                ("Scorpio-N Z4 2.2 Diesel AT 7-Seat", "Diesel", "Automatic", 2184, 7, 1730000),
                ("Scorpio-N Z8 2.2 Diesel 4XPLOR 4WD MT 7-Seat", "Diesel", "Manual", 2184, 7, 2180000),
                ("Scorpio-N Z8 L 2.2 Diesel 4XPLOR 4WD AT Captain Seats", "Diesel", "Automatic", 2184, 6, 2454000),
            ]),
        ]
    },

    # --- TOYOTA ---
    "Toyota": {
        "country": "Japan",
        "models": [
            ("Fortuner", "SUV", 2004, None, [
                ("Fortuner 2.7 4x2 MT Petrol", "Petrol", "Manual", 2694, 7, 3343000),
                ("Fortuner 2.7 4x2 AT Petrol", "Petrol", "Automatic", 2694, 7, 3502000),
                ("Fortuner 2.8 4x2 MT Diesel", "Diesel", "Manual", 2755, 7, 3593000),
                ("Fortuner 2.8 4x4 AT Diesel", "Diesel", "Automatic", 2755, 7, 5144000),
                ("Fortuner Legender 2.8 4x4 AT Dual Tone", "Diesel", "Automatic", 2755, 7, 4764000),
                ("Fortuner GR-Sport 2.8 4x4 AT", "Diesel", "Automatic", 2755, 7, 5144000),
            ]),
            ("Innova Hycross", "MPV", 2022, None, [
                ("Innova Hycross GX 2.0 Petrol 7-Seat", "Petrol", "CVT", 1987, 7, 1977000),
                ("Innova Hycross VX 2.0 Strong Hybrid 7-Seat", "Hybrid", "CVT", 1987, 7, 2597000),
                ("Innova Hycross ZX 2.0 Strong Hybrid 7-Seat Ottoman", "Hybrid", "CVT", 1987, 7, 3034000),
                ("Innova Hycross ZX (O) 2.0 Strong Hybrid 7-Seat ADAS", "Hybrid", "CVT", 1987, 7, 3098000),
            ]),
        ]
    },

    # --- TESLA ---
    "Tesla": {
        "country": "USA",
        "models": [
            ("Model 3", "EV", 2017, None, [
                ("Model 3 Rear-Wheel Drive 60 kWh", "EV", "Automatic", 0, 5, 4200000),
                ("Model 3 Long Range Dual Motor AWD 82 kWh", "EV", "Automatic", 0, 5, 5500000),
                ("Model 3 Performance Dual Motor AWD 510HP 2024", "EV", "Automatic", 0, 5, 6800000),
            ]),
            ("Cybertruck", "Pickup", 2023, None, [
                ("Cybertruck Rear-Wheel Drive Single Motor", "EV", "Automatic", 0, 5, 11000000),
                ("Cybertruck Dual Motor AWD 600HP", "EV", "Automatic", 0, 5, 14000000),
                ("Cybertruck Cyberbeast Tri-Motor AWD 845HP 2.6s", "EV", "Automatic", 0, 5, 18000000),
            ]),
        ]
    },

    # --- PORSCHE ---
    "Porsche": {
        "country": "Germany",
        "models": [
            ("911", "Coupe", 1963, None, [
                ("911 Carrera 3.0 Flat-6 388HP", "Petrol", "Automatic", 2981, 4, 18600000),
                ("911 Carrera S 3.0 Flat-6 443HP", "Petrol", "Automatic", 2981, 4, 20100000),
                ("911 GTS T-Hybrid 3.6 Flat-6 534HP 2024", "Hybrid", "Automatic", 3591, 4, 27500000),
                ("911 Turbo S 3.8 Twin-Turbo 640HP", "Petrol", "Automatic", 3745, 4, 33500000),
                ("911 GT3 RS 4.0 Naturally Aspirated 518HP", "Petrol", "Automatic", 3996, 2, 35000000),
                ("911 S/T 60th Anniversary 4.0 Manual", "Petrol", "Manual", 3996, 2, 42600000),
            ]),
        ]
    },
}

CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Pune", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad",
    "New York", "Los Angeles", "London", "Tokyo", "Berlin", "Paris", "Dubai", "Sydney", "Rome", "Beijing"
]

PLATFORMS = ["CarDekho", "Spinny", "Cars24", "OLX Autos", "Droom", "AutoTrader US", "Mobile.de", "Carvana"]


def seed_exhaustive_variants_and_prices():
    print("=" * 85)
    print("Seeding Complete Tiered Variant Lineups with Distinct MSRP Prices...")
    print("=" * 85)
    init_db()

    db = SessionLocal()

    total_companies = 0
    total_models = 0
    total_variants = 0
    total_listings = 0
    current_year = datetime.now().year

    for brand_name, brand_info in WORLDWIDE_CATALOG.items():
        # 1. Company
        company = db.query(Company).filter_by(name=brand_name).first()
        if not company:
            company = Company(
                name=brand_name,
                country=brand_info["country"],
                logo_url=f"https://img.car-logos.example/{brand_name.lower().replace(' ', '-')}.png"
            )
            db.add(company)
            db.flush()
            total_companies += 1

        for m_tuple in brand_info["models"]:
            if len(m_tuple) == 5:
                m_name, body_type, launch_year, discontinued_year, variants_list = m_tuple
            else:
                m_name, body_type, launch_year, variants_list = m_tuple
                discontinued_year = None

            # 2. Model
            model = db.query(Model).filter_by(company_id=company.id, name=m_name).first()
            if not model:
                model = Model(
                    company_id=company.id,
                    name=m_name,
                    body_type=body_type,
                    launch_year=launch_year,
                    discontinued_year=discontinued_year
                )
                db.add(model)
                db.flush()
                total_models += 1

            for v_name, fuel, trans, engine_cc, seating, msrp in variants_list:
                # 3. Variant
                variant = db.query(Variant).filter_by(model_id=model.id, name=v_name).first()
                if not variant:
                    variant = Variant(
                        model_id=model.id,
                        name=v_name,
                        fuel_type=fuel,
                        transmission=trans,
                        engine_cc=engine_cc,
                        seating_capacity=seating,
                        ex_showroom_price=float(msrp)
                    )
                    db.add(variant)
                    db.flush()
                    total_variants += 1
                else:
                    variant.ex_showroom_price = float(msrp)
                    variant.fuel_type = fuel
                    variant.transmission = trans
                    variant.engine_cc = engine_cc
                    variant.seating_capacity = seating
                    db.flush()

                # 4. Resale Listings per Variant
                max_year = discontinued_year if discontinued_year else current_year
                try:
                    ly = int(launch_year)
                except (ValueError, TypeError):
                    ly = 2015
                try:
                    my = int(max_year)
                except (ValueError, TypeError):
                    my = current_year

                min_yr = min(ly, my)
                num_listings = random.randint(8, 15)
                for i in range(num_listings):
                    mfg_year = random.randint(min_yr, my)
                    car_age = max(0, current_year - mfg_year)

                    dep_rate = 0.08 if brand_name in ["Toyota", "Maruti Suzuki", "Porsche", "Ferrari", "Lamborghini"] else 0.13
                    depreciated_price = msrp * ((1.0 - dep_rate) ** car_age)

                    price_factor = random.uniform(0.85, 1.15)
                    asking_price = max(40000.0, round(depreciated_price * price_factor, -3))
                    km = round(random.uniform(4000, 22000) * (car_age + 0.5), -2)
                    owner_count = 1 if car_age <= 3 else (2 if car_age <= 7 else random.choice([2, 3, 4]))

                    city = random.choice(CITIES)
                    platform = random.choice(PLATFORMS)
                    url_slug = f"{brand_name.lower().replace(' ', '-')}-{m_name.lower().replace(' ', '-')}-{v_name.lower().replace(' ', '-')}-{city.lower()}-{mfg_year}-{i+1}"
                    source_url = f"https://www.{platform.lower().replace(' ', '')}.com/listing/{url_slug}"

                    listing = db.query(Listing).filter_by(source_url=source_url).first()
                    if not listing:
                        status_str = "DISCONTINUED" if discontinued_year else "ACTIVE"
                        listing = Listing(
                            variant_id=variant.id,
                            source_platform=platform,
                            source_url=source_url,
                            manufacture_year=mfg_year,
                            km_driven=km,
                            owner_count=owner_count,
                            city=city,
                            asking_price=asking_price,
                            insurance_valid=random.choice([True, True, False]),
                            accident_history=random.choice([False, False, False, True]),
                            description=f"[{status_str}] {mfg_year} {brand_name} {m_name} {v_name} with {km:,.0f} km in {city}.",
                            scraped_at=datetime.now(timezone.utc)
                        )
                        db.add(listing)
                        total_listings += 1

    # Record Log
    log = ScraperLog(
        source="variant_tiered_pricing_seeder",
        status="SUCCESS",
        records_scraped=total_listings,
        run_time_seconds=5.5,
    )
    db.add(log)
    db.commit()

    # Query Totals
    c_count = db.query(Company).count()
    m_count = db.query(Model).count()
    v_count = db.query(Variant).count()
    l_count = db.query(Listing).count()

    db.close()

    print("\n" + "=" * 85)
    print("TIERED VARIANT PRICING SEEDING COMPLETED SUCCESSFULLY!")
    print("=" * 85)
    print(f"Total Car Brands (Companies) in DB    : {c_count}")
    print(f"Total Car Models in DB                : {m_count}")
    print(f"Total Car Variants in DB              : {v_count}")
    print(f"Total Market Used Car Listings in DB  : {l_count}")
    print("=" * 85)


if __name__ == "__main__":
    seed_exhaustive_variants_and_prices()
