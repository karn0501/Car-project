# Used Car Price Prediction Software — Project Documentation

**Version:** 1.0
**Type:** Advanced-level, production-grade ML/DL system
**Target Hardware:** GPU-accelerated training (RTX 4060 or equivalent, 8GB VRAM)

---

## 1. Project Overview

This project is an advanced used-car price prediction platform that combines:

- A **self-updating, live database** of car companies, models, and variants (built via web scraping, not static datasets)
- A **machine learning ensemble engine** for accurate price prediction
- **Deep learning modules** for image-based condition assessment, text-based listing quality scoring, and price trend forecasting
- **Explainable AI** so predictions are transparent, not a black box
- A full-stack web application (API + frontend) to serve predictions to end users

The goal is not a beginner "train one model on a CSV" project — it is a system architected the way real platforms (e.g., CarDekho, Cars24, Spinny) approach the problem: live data, ensemble modeling, explainability, and continuous retraining.

---

## 2. Objectives

1. Maintain an always up-to-date database of every company → model → variant available in the market, along with their specifications and prices.
2. Predict the resale price of any used car as accurately as possible using structured data, image data, and text data.
3. Provide transparent, explainable predictions (why a price was given, not just a number).
4. Build a scalable data pipeline that keeps itself current without manual re-collection.
5. Deliver the prediction through a clean, usable web application.

---

## 3. Data Strategy

### 3.1 Why Ready-Made Datasets Are Not Enough

Public datasets (Kaggle, etc.) are useful for prototyping and validating the ML approach early, but they go stale quickly, don't cover every brand/model/variant, and don't reflect real-time market prices. For a genuinely advanced system, **live scraping is required** as the primary data source.

### 3.2 Data Sources

**New car specs & official prices (variant-level):**
- CarDekho, CarWale, Zigwheels, 91wheels

**Used car live listings (actual resale prices):**
- Cars24, Spinny, CarDekho-Used, OLX Autos, Droom, Quikr

**Government reference data (optional, for registration trends):**
- Vahan Dashboard (parivahan.gov.in)

> **Note:** Always check each site's `robots.txt` and terms of service, use rate-limiting/throttling, and prefer official APIs where available. Respectful, throttled scraping avoids IP bans and legal issues.

### 3.3 Initial Bootstrapping (Prototyping Phase)

Before the live pipeline is fully built, use existing datasets to validate the ML approach:
- Indian used car datasets with brand, model, year, km driven, transmission, owner, fuel type, and price (~9,500+ entries)
- Larger Indian used car datasets with 7,000+ rows and ~29 columns of variant-level detail
- Multi-brand Indian car market datasets covering major brands with specs and pricing

These are for **early validation only** — the production system must transition to the live scraping pipeline.

---

## 4. Database Design

### 4.1 Schema (Relational — PostgreSQL)

```
companies
  id, name, logo_url, country

models
  id, company_id (FK), name, body_type, launch_year, discontinued_year

variants
  id, model_id (FK), name, fuel_type, transmission, engine_cc,
  seating_capacity, ex_showroom_price, launch_date

listings
  id, variant_id (FK), source_platform, source_url,
  manufacture_year, km_driven, owner_count, city,
  asking_price, insurance_valid, accident_history,
  scraped_at, is_active

price_history
  id, variant_id (FK), city, avg_price, date

scraper_logs
  id, source, status, records_scraped, run_time
```

### 4.2 Hierarchy Logic

```
Company → Model → Variant → Listings (real resale instances) → Price History (time series)
```

This structure allows variant-level accuracy (e.g., Swift LXI vs Swift ZXI(O) have very different resale values, not just "Swift").

---

## 5. Feature List

### 5.1 Core Features (Must-Have)
- Brand, Model, Variant
- Manufacturing Year / Age
- KM Driven
- Fuel Type (Petrol / Diesel / CNG / Electric / Hybrid)
- Transmission (Manual / Automatic)
- Owner Count (1st / 2nd / 3rd+)
- Body Type (Hatchback / Sedan / SUV / MPV)
- Seating Capacity
- Engine Capacity (cc)
- Mileage (kmpl)
- Location / City (resale prices vary significantly by city)

### 5.2 Advanced Features (For High Accuracy)
- Insurance status (valid/expired) and type
- Accident history / damage registration
- RTO / registration state (interstate cars often have lower resale value)
- Service history (authorized service center records)
- Owner type (individual vs. commercial/fleet)
- Color
- Original showroom price (for depreciation baseline calculation)
- Market demand index (current popularity of the model)
- Seasonal/festive timing effects
- Loan/EMI closure status
- NCAP safety rating

### 5.3 Derived/Engineered Features
- Car Age = Current Year − Manufacture Year
- Price per KM driven
- Brand average depreciation rate
- City price index

---

## 6. Machine Learning & Deep Learning Architecture

### 6.1 Core Principle

Used car pricing is a **tabular data problem**. For tabular data, gradient boosting models consistently outperform deep neural networks unless the dataset reaches hundreds of thousands of rows. Deep learning is used strategically for specific sub-tasks (images, text, sequences), not as the core predictor.

### 6.2 Core Prediction Engine — Stacked Ensemble

```
Base Models:  CatBoost (GPU) + XGBoost (GPU hist mode) + LightGBM (GPU)
                        ↓
Meta-Model:   Linear Regression (combines base model outputs)
                        ↓
Final Price Prediction
```

- **CatBoost** — handles categorical features (brand, model, fuel type, city) natively without manual encoding; strong fit for this dataset type.
- **XGBoost / LightGBM** — GPU-accelerated training for speed and additional model diversity in the ensemble.
- **Hyperparameter tuning** — Optuna, run with GPU-parallel trials for faster search.

### 6.3 Deep Learning Modules (Where GPU Is Genuinely Used)

| Module | Purpose | Technique |
|---|---|---|
| Image condition scoring | Detect dents/scratches/rust from uploaded photos, adjust price | CNN transfer learning (EfficientNet-B0 / ResNet50) |
| Listing description scoring | Extract quality signal from free-text descriptions ("well maintained", "single owner") | DistilBERT fine-tuned (LoRA for efficiency) |
| Categorical entity embeddings | Learn relationships between brands/models beyond one-hot encoding | Small embedding neural network (PyTorch) |
| Price trend forecasting | Predict future market price direction over time | LSTM / GRU on `price_history` table |

These modules produce **additional features** that feed into the core ensemble model, rather than replacing it.

### 6.4 Explainability

- **SHAP (SHapley Additive Explanations)** applied to the ensemble output to show exactly why a price was predicted (e.g., "-₹20,000 due to KM driven", "+₹10,000 due to brand value").
- This is a defining feature of a professional-grade system and should not be skipped.

---

## 7. System Architecture (End-to-End)

```
┌─────────────────────┐
│  Scrapy / Playwright │  → distributed spiders across multiple sources
│  Spiders             │
└──────────┬───────────┘
           ↓
┌─────────────────────┐
│  Kafka / RabbitMQ     │  → decouples scraping from processing
│  Message Queue        │
└──────────┬───────────┘
           ↓
┌─────────────────────┐
│  Validation Layer      │  → Pydantic schema checks, deduplication (rapidfuzz)
└──────────┬───────────┘
           ↓
┌─────────────────────┐
│  Raw Data Lake         │  → MongoDB / AWS S3
└──────────┬───────────┘
           ↓
      Apache Airflow (ETL scheduling)
           ↓
┌─────────────────────┐
│  PostgreSQL            │  → clean, structured, relational data
└──────────┬───────────┘
           ↓
┌─────────────────────┐
│  Feature Store          │  → precomputed ML features
└──────────┬───────────┘
           ↓
┌─────────────────────┐
│  ML/DL Ensemble         │  → CatBoost + XGBoost + LightGBM + CNN/BERT/LSTM features
│  + SHAP Explainability   │
└──────────┬───────────┘
           ↓
┌─────────────────────┐
│  FastAPI Backend        │  → serves predictions via REST API
└──────────┬───────────┘
           ↓
┌─────────────────────┐
│  React/Next.js Frontend │  → user interface
└─────────────────────┘
```

### 7.1 Supporting Infrastructure
- **Proxy rotation** (BrightData/ScraperAPI or self-hosted pool) — avoid IP bans during scraping
- **Anti-detection** — `playwright-stealth`, randomized user-agents, human-like delays
- **MLflow** — model versioning and experiment tracking
- **Redis + Elasticsearch** — fast filtering and search across listings
- **Docker + Kubernetes** — containerized, scalable deployment
- **Grafana + Prometheus** — monitoring scraper health and API uptime
- **CI/CD** — GitHub Actions for automated deployment

---

## 8. Tech Stack Summary

| Layer | Technology |
|---|---|
| Scraping | Scrapy, Playwright |
| Queue | Kafka / RabbitMQ |
| Raw storage | MongoDB / AWS S3 |
| Structured storage | PostgreSQL |
| Orchestration | Apache Airflow |
| ML Core | CatBoost, XGBoost, LightGBM |
| DL Modules | PyTorch/TensorFlow (CNN, BERT, LSTM) |
| Tuning | Optuna |
| Explainability | SHAP |
| Model tracking | MLflow |
| Backend API | FastAPI |
| Frontend | React.js / Next.js |
| Caching/Search | Redis, Elasticsearch |
| Deployment | Docker, Kubernetes, AWS/GCP |
| Monitoring | Grafana, Prometheus |

---

## 9. Development Roadmap

### Phase 1 — Data Pipeline Foundation

**Goal:** Get real, structured car data flowing from one live source into your own database. This is the foundation everything else depends on, so it should not be rushed.

1. Set up the local environment: Python 3.10+, PostgreSQL installed and running, a virtual environment, and `scrapy` installed.
2. Design and create the PostgreSQL tables exactly as defined in Section 4 (`companies`, `models`, `variants`, `listings`, `price_history`, `scraper_logs`) using a migration tool such as Alembic so schema changes are trackable.
3. Pick **one** source to start with (e.g., CarDekho used-car listings for a single city). Inspect the page structure, check `robots.txt`, and confirm which fields are visible without login (price, year, km, fuel type, transmission, owner, city).
4. Write a single Scrapy spider that crawls listing pages, extracts these fields, and yields structured items (use Scrapy `Item` classes, not raw dicts, for validation).
5. Add a pipeline in Scrapy (`ItemPipeline`) that writes each scraped item directly into the `listings` table via `psycopg2` or `SQLAlchemy`, linking it to the correct `variant_id` (create a matching `company`/`model`/`variant` row if it doesn't exist yet).
6. Add basic rate-limiting (`DOWNLOAD_DELAY`, `AUTOTHROTTLE_ENABLED` in Scrapy settings) so the scraper behaves respectfully.
7. Run the spider, confirm rows appear correctly in PostgreSQL, and manually spot-check 10–15 records against the live site for accuracy.
8. **Add a basic anomaly filter** in the validation step: reject or flag listings with impossible values (e.g., price far below/above the typical range for that variant, negative km, future manufacture year, test/dummy listings). A simple z-score or IQR-based check per variant is enough at this stage — this prevents bad data from ever entering your training set.
9. **Write basic unit tests** for the scraper's parsing functions and the database-write pipeline (e.g., using `pytest`), so a future site-layout change breaks a test loudly instead of silently corrupting data.

**Deliverable:** A working scraper that reliably populates your database with a few hundred to a few thousand real, validated listings from one source, with correct schema relationships and basic tests in place.

---

### Phase 2 — Baseline Model

**Goal:** Prove the core idea works before investing in complexity. This phase should be simple and fast.

1. Export the current PostgreSQL data into a Pandas DataFrame using `pandas.read_sql`.
2. Do basic exploratory data analysis (EDA) in a Jupyter notebook: check distributions of price, km driven, and year; look for obvious outliers (e.g., 2 lakh km on a 1-year-old car) and remove or cap them.
3. Engineer the basic derived features: car age, price-per-km, and one-hot or target encoding for city (if not using CatBoost yet).
4. Split the data into train/test sets (e.g., 80/20), keeping a fixed random seed for reproducibility.
5. Train a single **CatBoost Regressor** (CPU is fine at this stage; dataset is still small) using only the core features from Section 5.1.
6. Evaluate using RMSE, MAE, and R² on the test set. Write these numbers down — this is your baseline to beat in every later phase.
7. Save the trained model (`model.save_model()`), and set up a simple script that loads it and predicts on a new sample input.

**Deliverable:** A single working model with a documented baseline accuracy score, saved to disk and loadable for testing.

---

### Phase 3 — Ensemble & Tuning

**Goal:** Push accuracy beyond the baseline using a stacked ensemble and proper tuning, now using your GPU.

1. Install GPU-enabled versions of XGBoost and LightGBM, and enable CatBoost's `task_type='GPU'`. Confirm each trains successfully on your RTX 4060 (check `nvidia-smi` during training to confirm GPU usage).
2. Train XGBoost (`tree_method='gpu_hist'`) and LightGBM (`device='gpu'`) separately on the same train/test split as Phase 2, and compare their individual RMSE/MAE/R² against the CatBoost baseline.
3. Build a stacking ensemble: generate out-of-fold predictions from CatBoost, XGBoost, and LightGBM using k-fold cross-validation, then train a simple Linear Regression meta-model on those predictions to produce the final output.
4. Install Optuna and define a tuning objective function for each base model (key parameters: learning rate, max depth, number of estimators, subsample ratio). Run Optuna studies with GPU-enabled trials, 50–100 trials per model is a reasonable start.
5. Retrain each base model with its best-found hyperparameters, rebuild the stacked ensemble, and re-evaluate. Confirm the ensemble beats every individual model and the Phase 2 baseline.
6. Log every experiment (model type, hyperparameters, resulting metrics) — a simple CSV log is fine for now, MLflow comes in Phase 8.
7. **Add confidence intervals, not just a point prediction.** Train quantile regression versions of CatBoost/LightGBM (e.g., predicting the 10th, 50th, and 90th percentile) so the final output can be shown as a realistic range (e.g., "₹5,80,000 – ₹6,30,000") instead of a single falsely-precise number. This is far more trustworthy to real users than one exact figure.

**Deliverable:** A tuned, stacked ensemble model that measurably outperforms the Phase 2 baseline, with logged experiment results and a price range (not just a point estimate) as output.

---

### Phase 4 — Explainability

**Goal:** Make every prediction interpretable, not just a number.

1. Install the `shap` library and confirm it supports your ensemble's base models (SHAP has native fast support for tree-based models like CatBoost/XGBoost/LightGBM via `TreeExplainer`).
2. For a given prediction, compute SHAP values for each base model and combine them (weighted by how much each contributes to the meta-model) to get an approximate feature contribution for the final ensemble output.
3. Build a simple function that takes one car's input features and returns a human-readable breakdown, e.g., `"Base value: ₹6,50,000 | KM driven: -₹20,000 | Age: -₹35,000 | Brand: +₹10,000 | Final: ₹6,05,000"`.
4. Generate a global SHAP summary plot across the whole test set to identify which features matter most overall — this is also useful for your own feature engineering decisions going forward.
5. Test explanations on 5–10 diverse cars to make sure the reasoning looks sensible and not contradictory.

**Deliverable:** A working explainability function that can be called alongside any prediction to show a clear, correct breakdown of contributing factors.

---

### Phase 5 — Deep Learning Modules (Image-Based Condition Detection)

**Goal:** Add your first real GPU-heavy deep learning component — this is where the RTX 4060 does genuine work.

1. Collect or download a car damage/condition image dataset (Kaggle has several "car damage detection" datasets with categories like scratch, dent, and no-damage).
2. Set up PyTorch or TensorFlow with CUDA support and confirm GPU is detected (`torch.cuda.is_available()`).
3. Load a pretrained CNN backbone (EfficientNet-B0 or ResNet50) and fine-tune it (transfer learning) on the damage dataset — freeze early layers, train the final layers first, then optionally unfreeze more layers for fine-tuning.
4. Train with a reasonable batch size (16–32 fits comfortably on 8GB VRAM), monitor validation accuracy/loss, and save the best checkpoint.
5. Build a simple inference function: input a car photo, output a "condition score" (e.g., a 0–1 scale or a category like minor/moderate/major damage).
6. Add this condition score as a new input feature to your Phase 3 ensemble, retrain the ensemble including this feature, and confirm it improves accuracy on cars where photos are available (this feature can be optional/null for listings without photos).

**Deliverable:** A trained CNN that scores car condition from an image, integrated as an additional feature into the main prediction ensemble.

---

### Phase 6 — Full Pipeline Automation

**Goal:** Turn your single-source, manually-run scraper into a live, self-updating, multi-source pipeline.

1. Write additional Scrapy spiders for 2–3 more sources (e.g., Spinny, OLX Autos, CarDekho variant/spec pages for new-car baseline prices).
2. Install and configure Apache Airflow. Create a DAG that runs each spider on a schedule (e.g., daily), followed by a data-validation and load task.
3. Add a deduplication step using fuzzy string matching (`rapidfuzz`) to merge listings that refer to the same model/variant but are spelled differently across sources (e.g., "Maruti Suzuki Swift" vs "Maruti Swift").
4. Add an incremental-scraping strategy: track `last_seen` timestamps per listing URL so you're not re-scraping unchanged data every run — only new or updated listings should be processed.
5. (Optional but recommended for true production scale) Introduce Kafka or RabbitMQ between the scrapers and the database loader, so scraping and processing are decoupled and can scale independently.
6. Set up proxy rotation and randomized delays/user-agents across all spiders to keep scraping stable long-term.
7. Add logging into the `scraper_logs` table so you can monitor how many records each source contributes per run, and catch failures early.
8. **Add alerting on pipeline failure.** If a source's HTML structure changes and a spider suddenly returns zero or near-zero records, send yourself a notification (email, Slack webhook, or even a simple cron-checked log flag) rather than letting the database silently go stale.
9. **Increase geographic granularity where data allows.** Once volume is high enough, move beyond city-level averages toward finer zones within a city (e.g., dealer-heavy areas vs peer-to-peer listings), since prices can meaningfully differ within the same city.

**Deliverable:** A scheduled, multi-source pipeline that keeps your database current automatically, with logging, deduplication, and failure alerting in place.

---

### Phase 7 — Advanced Modules (NLP & Forecasting)

**Goal:** Add the remaining deep learning components for a fully-rounded advanced system.

1. **Description scoring:** Collect listing description text where available. Fine-tune a DistilBERT model (using LoRA/PEFT to keep VRAM usage manageable) to output a "listing quality score" — this can be bootstrapped using simple heuristics initially (keyword presence like "single owner", "showroom condition") as weak labels, then refined.
2. Integrate the description score as another feature into the ensemble, retrain, and evaluate the improvement.
3. **Price trend forecasting:** Using the `price_history` table (populated over time as your pipeline runs), build a time-series dataset per model/variant/city.
4. Train an LSTM or GRU model on this time-series data to forecast how a given variant's average price is likely to move over the next few months.
5. Expose this as a separate "price trend" feature in the application (e.g., a graph showing predicted price direction), distinct from the individual car prediction.
6. Validate forecasts against held-out recent months of real data to sanity-check the trend model isn't just extrapolating noise.
7. **Add macroeconomic and seasonal signals** as extra time-series inputs to the LSTM/GRU: fuel price trends, interest rate movements (affects EMI affordability, which affects used-car demand), and festive-season markers (e.g., demand typically rises around Diwali). These genuinely move used-car prices and improve forecast quality.

**Deliverable:** Two additional working modules — a text-based quality scorer and a time-series price trend forecaster (incorporating macroeconomic signals) — both integrated or exposed alongside the main prediction.

---

### Phase 8 — Productionization

**Goal:** Turn the trained models and pipeline into a real, usable, deployable application.

1. Set up **MLflow** to track every model version going forward (parameters, metrics, artifacts) — retroactively log your best Phase 3/5/7 models so you have a clean version history from this point on.
2. Build a **FastAPI** backend exposing endpoints such as `/predict` (core price prediction with SHAP breakdown), `/upload-image` (condition scoring), and `/trend/{variant_id}` (price forecast).
3. Containerize the backend, database, and scraper components separately using Docker, and define a `docker-compose.yml` for local orchestration (Kubernetes only if you need multi-machine scaling later).
4. Build the **React/Next.js** frontend: a form for car details, an image upload component, a results page showing predicted price with the SHAP explanation, and a trend graph.
5. Set up GitHub Actions for CI/CD — automatically test and deploy on push to main.
6. Add basic monitoring: Prometheus metrics for API latency/uptime, and a Grafana dashboard; optionally add drift detection (e.g., Evidently AI) to flag when live predictions start deviating from real observed prices.
7. Deploy: backend and database to a cloud provider (AWS/GCP/Render/Railway), frontend to Vercel, and confirm the full flow works end-to-end from a live URL.
8. **Add a "comparable listings" panel** to the results page: alongside the predicted price/range, show 3–5 real, similar listings from your database so users can visually sanity-check the number. This builds trust far faster than a number alone.
9. **Auto-generate a downloadable PDF valuation report** per prediction (predicted price/range, SHAP breakdown, comparable listings, trend chart). This single feature is what separates a "student project" feel from a "real product" feel.
10. **Add a feedback field** ("What did you actually sell/buy this car for?") that writes back into a `feedback` table. This ground-truth data becomes valuable for future retraining and for measuring your real-world error rate, not just offline test-set metrics.
11. **Secure the API** with rate limiting and API key authentication (FastAPI supports this via middleware/dependencies) before making it public, so it can't be abused or scraped itself.
12. **Write a basic privacy policy** and avoid storing more user-submitted personal data than necessary, especially if you plan to grow this beyond a portfolio project.

**Deliverable:** A fully deployed, publicly accessible application with a tracked model history, monitoring, an automated deployment pipeline, comparable listings, PDF reports, a feedback loop, and basic API security.

---

### Phase 9 — Market Growth Features (Post-MVP)

**Goal:** Once the core product is accurate and stable, these features turn it from a working tool into something people would actually adopt, return to, or pay for. Treat this phase as optional/incremental — pick items based on your actual goals (portfolio vs. real product).

1. **User accounts** — Let users sign up, save past valuations, and track how a specific car's value changes over time. Simple JWT-based auth via FastAPI is enough to start.
2. **Conversational/chatbot interface** — Let users type a natural query like *"2019 Swift VXI, 40,000 km, Ahmedabad — what's it worth?"* and have an LLM parse this into structured input for your existing pipeline, instead of forcing a long form. This significantly improves usability.
3. **API / B2B offering** — Package your `/predict` endpoint as a paid API product for dealerships, insurance companies, or classifieds sites. This is a realistic monetization path once your accuracy and uptime are proven.
4. **Mobile access** — At minimum, make the frontend a installable PWA (Progressive Web App); a full native app is a later step once there's validated demand.
5. **Seller/dealer trust scoring** — If you ever allow listings directly on your own platform, score sellers based on response rate, document verification, and transaction history to build marketplace trust.
6. **A/B testing framework** — Before fully replacing a production model with a newer version, route a portion of live traffic to each and compare real-world error rates before a full rollout.

**Deliverable:** A set of growth-oriented features layered on top of the stable core product, chosen based on whether the goal is a portfolio showcase or an actual market-facing product.

---

## 10. Evaluation Metrics

- **RMSE (Root Mean Squared Error)** — penalizes large prediction errors
- **MAE (Mean Absolute Error)** — average prediction error magnitude
- **R² Score** — how much price variance the model explains
- **SHAP-based feature importance** — qualitative validation of model reasoning
- **Drift monitoring** — track whether live predictions deviate from real market prices over time (e.g., using Evidently AI)

---

## 11. Key Design Principles

1. **Live data over static datasets** — the database must self-update via scheduled scraping, not remain frozen.
2. **Boosting over deep learning for core prediction** — gradient boosting models (CatBoost/XGBoost/LightGBM) are the right tool for structured/tabular data.
3. **Deep learning where it adds real value** — images, text, and time-series are the legitimate use cases for GPU-heavy training in this project.
4. **Explainability is not optional** — every prediction should be justifiable via SHAP.
5. **Pipeline automation** — the system should require minimal manual intervention to stay current (Airflow-driven retraining and rescraping).
6. **Respectful scraping** — rate-limiting, robots.txt compliance, and proxy rotation to avoid legal/technical issues.

---

## 12. Next Steps

The recommended build order is: scraper → baseline model → ensemble (with confidence intervals) → explainability → image module → pipeline automation (with alerting) → NLP/time-series modules (with macroeconomic signals) → full-stack deployment (with reports, comparables, and feedback loop) → optional growth features. Each phase should be functional and tested before moving to the next. Phases 1–8 form the complete, accurate, production-grade core product; Phase 9 is only needed if the goal extends beyond a portfolio project into a real market-facing product.

---

## 13. Other Things Worth Keeping In Mind

A few remaining items that don't belong to any single phase but matter throughout the project's life:

- **Legal/IP awareness of scraping** — Terms of service differ by site; some explicitly prohibit scraping. Treat this as an ongoing risk to monitor, not a one-time check in Phase 1.
- **Cloud cost management** — GPU training is local (free), but cloud hosting (database, API, scheduled scraping jobs) has ongoing costs. Track usage early so hosting costs don't surprise you once traffic grows.
- **Documentation discipline** — Keep this document (or a living version of it, e.g., a README/wiki) updated as the system evolves — a project this complex becomes hard to maintain or hand off without it.
- **Security review before public launch** — Beyond API rate-limiting, review for SQL injection risk (use parameterized queries/ORM), secrets management (never hardcode API keys/DB credentials — use environment variables or a secrets manager), and dependency vulnerability scanning.
