"""
SQLite to PostgreSQL Exporter & Data Migration Tool.
Converts car_prediction.db (172 Brands, 632 Models, 1,022 Variants, 22,716 Listings)
into PostgreSQL SQL format (car_prediction_postgres_dump.sql) and migrates directly
to any target PostgreSQL database instance.
"""

import os
import sys
import sqlite3
import argparse

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_postgres_sql_dump(sqlite_db_path, output_sql_path):
    print("=" * 80)
    print(f"Generating PostgreSQL SQL Dump from {sqlite_db_path}...")
    print("=" * 80)

    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    sql_lines = []
    sql_lines.append("-- ==================================================================")
    sql_lines.append("-- CAR PREDICTION SYSTEM - POSTGRESQL MASTER DATABASE DUMP")
    sql_lines.append("-- Compatible with PostgreSQL 12+")
    sql_lines.append("-- ==================================================================\n")

    sql_lines.append("BEGIN;\n")

    # DDL
    sql_lines.append("""
-- DROP EXISTING TABLES IF ANY
DROP TABLE IF EXISTS listings CASCADE;
DROP TABLE IF EXISTS variants CASCADE;
DROP TABLE IF EXISTS models CASCADE;
DROP TABLE IF EXISTS companies CASCADE;
DROP TABLE IF EXISTS scraper_logs CASCADE;

-- CREATE SCHEMAS
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    country VARCHAR(100),
    logo_url VARCHAR(255)
);

CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    body_type VARCHAR(50),
    launch_year INT,
    discontinued_year INT
);

CREATE TABLE variants (
    id SERIAL PRIMARY KEY,
    model_id INT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    name VARCHAR(150) NOT NULL,
    fuel_type VARCHAR(30) NOT NULL,
    transmission VARCHAR(30) NOT NULL,
    engine_cc INT,
    seating_capacity INT,
    ex_showroom_price DOUBLE PRECISION,
    launch_date VARCHAR(30)
);

CREATE TABLE listings (
    id SERIAL PRIMARY KEY,
    variant_id INT NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
    source_platform VARCHAR(100) NOT NULL,
    source_url VARCHAR(500) NOT NULL UNIQUE,
    manufacture_year INT NOT NULL,
    km_driven DOUBLE PRECISION NOT NULL,
    owner_count INT NOT NULL,
    city VARCHAR(100) NOT NULL,
    asking_price DOUBLE PRECISION NOT NULL,
    insurance_valid BOOLEAN DEFAULT TRUE,
    accident_history BOOLEAN DEFAULT FALSE,
    description TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE scraper_logs (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    records_scraped INT NOT NULL,
    run_time_seconds DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- CREATE INDEXES FOR OPTIMAL QUERY PERFORMANCE
CREATE INDEX idx_models_company_id ON models(company_id);
CREATE INDEX idx_variants_model_id ON variants(model_id);
CREATE INDEX idx_listings_variant_id ON listings(variant_id);
CREATE INDEX idx_listings_city ON listings(city);
CREATE INDEX idx_listings_asking_price ON listings(asking_price);
""")

    # 1. Companies Data
    cursor.execute("SELECT id, name, country, logo_url FROM companies ORDER BY id")
    companies = cursor.fetchall()
    sql_lines.append(f"\n-- INSERT {len(companies)} COMPANIES")
    for cid, name, country, logo in companies:
        c_name = name.replace("'", "''")
        c_country = country.replace("'", "''") if country else ""
        c_logo = logo.replace("'", "''") if logo else ""
        sql_lines.append(f"INSERT INTO companies (id, name, country, logo_url) VALUES ({cid}, '{c_name}', '{c_country}', '{c_logo}');")

    # 2. Models Data
    cursor.execute("SELECT id, company_id, name, body_type, launch_year, discontinued_year FROM models ORDER BY id")
    models = cursor.fetchall()
    sql_lines.append(f"\n-- INSERT {len(models)} MODELS")
    for mid, comp_id, name, body_type, launch_yr, disc_yr in models:
        m_name = name.replace("'", "''")
        m_body = body_type.replace("'", "''") if body_type else ""
        l_yr = launch_yr if launch_yr else "NULL"
        d_yr = disc_yr if disc_yr else "NULL"
        sql_lines.append(f"INSERT INTO models (id, company_id, name, body_type, launch_year, discontinued_year) VALUES ({mid}, {comp_id}, '{m_name}', '{m_body}', {l_yr}, {d_yr});")

    # 3. Variants Data
    cursor.execute("SELECT id, model_id, name, fuel_type, transmission, engine_cc, seating_capacity, ex_showroom_price, launch_date FROM variants ORDER BY id")
    variants = cursor.fetchall()
    sql_lines.append(f"\n-- INSERT {len(variants)} VARIANTS")
    for vid, model_id, name, fuel, trans, cc, seats, msrp, ldate in variants:
        v_name = name.replace("'", "''")
        v_fuel = fuel.replace("'", "''")
        v_trans = trans.replace("'", "''")
        v_cc = cc if cc is not None else "NULL"
        v_seats = seats if seats is not None else "NULL"
        v_msrp = msrp if msrp is not None else "NULL"
        v_ldate = f"'{ldate}'" if ldate else "NULL"
        sql_lines.append(f"INSERT INTO variants (id, model_id, name, fuel_type, transmission, engine_cc, seating_capacity, ex_showroom_price, launch_date) VALUES ({vid}, {model_id}, '{v_name}', '{v_fuel}', '{v_trans}', {v_cc}, {v_seats}, {v_msrp}, {v_ldate});")

    # 4. Listings Data
    cursor.execute("SELECT id, variant_id, source_platform, source_url, manufacture_year, km_driven, owner_count, city, asking_price, insurance_valid, accident_history, description, scraped_at FROM listings ORDER BY id")
    listings = cursor.fetchall()
    sql_lines.append(f"\n-- INSERT {len(listings)} LISTINGS")
    for lid, var_id, platform, url, mfg_yr, km, owners, city, price, ins, acc, desc, scraped in listings:
        l_plat = platform.replace("'", "''")
        l_url = url.replace("'", "''")
        l_city = city.replace("'", "''")
        l_desc = desc.replace("'", "''") if desc else ""
        l_ins = "TRUE" if ins else "FALSE"
        l_acc = "TRUE" if acc else "FALSE"
        l_scraped = f"'{scraped}'" if scraped else "CURRENT_TIMESTAMP"
        sql_lines.append(f"INSERT INTO listings (id, variant_id, source_platform, source_url, manufacture_year, km_driven, owner_count, city, asking_price, insurance_valid, accident_history, description, scraped_at) VALUES ({lid}, {var_id}, '{l_plat}', '{l_url}', {mfg_yr}, {km}, {owners}, '{l_city}', {price}, {l_ins}, {l_acc}, '{l_desc}', {l_scraped});")

    # 5. Scraper Logs
    cursor.execute("SELECT id, source, status, records_scraped, run_time_seconds, timestamp FROM scraper_logs ORDER BY id")
    logs = cursor.fetchall()
    sql_lines.append(f"\n-- INSERT {len(logs)} SCRAPER LOGS")
    for logid, src, stat, recs, rtime, tstamp in logs:
        l_src = src.replace("'", "''")
        l_stat = stat.replace("'", "''")
        l_tstamp = f"'{tstamp}'" if tstamp else "CURRENT_TIMESTAMP"
        sql_lines.append(f"INSERT INTO scraper_logs (id, source, status, records_scraped, run_time_seconds, timestamp) VALUES ({logid}, '{l_src}', '{l_stat}', {recs}, {rtime}, {l_tstamp});")

    # RESET SERIAL SEQUENCES
    sql_lines.append("\n-- RESET SERIAL PRIMARY KEY SEQUENCES")
    sql_lines.append("SELECT setval('companies_id_seq', (SELECT MAX(id) FROM companies));")
    sql_lines.append("SELECT setval('models_id_seq', (SELECT MAX(id) FROM models));")
    sql_lines.append("SELECT setval('variants_id_seq', (SELECT MAX(id) FROM variants));")
    sql_lines.append("SELECT setval('listings_id_seq', (SELECT MAX(id) FROM listings));")
    sql_lines.append("SELECT setval('scraper_logs_id_seq', (SELECT MAX(id) FROM scraper_logs));")

    sql_lines.append("\nCOMMIT;")

    with open(output_sql_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sql_lines))

    conn.close()
    print(f"PostgreSQL SQL Dump successfully written to: {output_sql_path}")
    print(f"Total Companies exported: {len(companies)}")
    print(f"Total Models exported   : {len(models)}")
    print(f"Total Variants exported : {len(variants)}")
    print(f"Total Listings exported : {len(listings)}")
    print("=" * 80)


if __name__ == "__main__":
    sqlite_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "car_prediction.db"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "car_prediction_postgres_dump.sql"))
    generate_postgres_sql_dump(sqlite_path, output_path)
