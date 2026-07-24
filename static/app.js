/* 
   AutoValuate AI — Application Logic & API Connectors
   Connects modern UI elements to FastAPI backend endpoints.
*/

document.addEventListener("DOMContentLoaded", () => {
    // 1. KM Driven Slider & Input Display Sync
    const kmInput = document.getElementById("km_driven");
    const kmDisplay = document.getElementById("km-val-display");

    if (kmInput && kmDisplay) {
        kmInput.addEventListener("input", (e) => {
            const val = parseInt(e.target.value) || 0;
            kmDisplay.innerText = val.toLocaleString("en-IN") + " km";
        });
    }

    // 1b. Dynamic Cascading Brand -> Model -> Variant Dropdowns
    const companySelect = document.getElementById("company_name");
    const modelSelect = document.getElementById("model_name");
    const variantSelect = document.getElementById("variant_name");

    let vehicleHierarchy = {
        "Maruti": {
            "Swift": ["VXi", "LXi", "ZXi", "ZXi Plus", "VDi", "ZDi"],
            "Baleno": ["Delta", "Sigma", "Zeta", "Alpha"],
            "Alto 800": ["LXi", "Std", "VXi"],
            "Dzire": ["VXi", "LXi", "ZXi", "ZXi Plus"],
            "Ertiga": ["VXi", "LXi", "ZXi", "ZXi Plus"],
            "Brezza": ["VXi", "LXi", "ZXi", "ZXi Plus"],
            "Wagon R": ["VXi", "LXi", "ZXi"]
        },
        "Hyundai": {
            "Creta": ["SX", "E", "EX", "S", "SX(O)"],
            "i20": ["Sportz", "Magna", "Asta", "Asta(O)"],
            "Verna": ["SX", "EX", "SX(O)"],
            "Venue": ["S", "E", "SX", "SX(O)"],
            "Grand i10": ["Sportz", "Era", "Magna", "Asta"]
        },
        "Tata": {
            "Nexon": ["XZ", "XE", "XM", "XZ+", "XZ+(O)"],
            "Harrier": ["XT", "XE", "XM", "XZ", "XZ+"],
            "Punch": ["Adventure", "Pure", "Accomplished", "Creative"],
            "Tiago": ["XT", "XE", "XM", "XZ", "XZ+"],
            "Safari": ["XT", "XE", "XM", "XZ", "XZ+"]
        },
        "Mahindra": {
            "Thar": ["LX", "AX", "AX(O)"],
            "Scorpio-N": ["Z4", "Z2", "Z6", "Z8", "Z8L"],
            "XUV700": ["AX5", "MX", "AX3", "AX7", "AX7L"],
            "Bolero": ["B6", "B4", "B6(O)"],
            "XUV300": ["W6", "W4", "W8", "W8(O)"]
        },
        "Honda": {
            "City": ["VX", "SV", "V", "ZX"],
            "Amaze": ["S", "E", "V", "VX"],
            "Civic": ["VX", "V", "ZX"],
            "WR-V": ["VX", "SV"]
        },
        "Toyota": {
            "Innova Crysta": ["VX", "GX", "ZX"],
            "Fortuner": ["4x2", "4x4", "Legender"],
            "Glanza": ["G", "E", "S", "V"]
        },
        "Kia": {
            "Seltos": ["HTX", "HTE", "HTK", "GTX", "X-Line"],
            "Sonet": ["HTX", "HTE", "HTK", "GTX+"],
            "Carens": ["Prestige", "Premium", "Luxury", "Luxury Plus"]
        },
        "Volkswagen": {
            "Polo": ["Highline", "Trendline", "Comfortline", "GT TSI"],
            "Vento": ["Highline", "Trendline", "Comfortline", "Highline Plus"],
            "Virtus": ["Dynamic Line", "Performance Line", "GT"],
            "Taigun": ["Highline", "Comfortline", "Topline", "GT"]
        }
    };

    async function initHierarchy() {
        try {
            const res = await fetch("/hierarchy");
            if (res.ok) {
                const data = await res.json();
                if (data && Object.keys(data).length > 0) {
                    vehicleHierarchy = data;
                }
            }
        } catch (e) {}

        populateCompanyDropdown();
    }

    function populateCompanyDropdown() {
        if (!companySelect) return;
        companySelect.innerHTML = "";
        Object.keys(vehicleHierarchy).forEach(brand => {
            const opt = document.createElement("option");
            opt.value = brand;
            opt.innerText = brand;
            companySelect.appendChild(opt);
        });

        companySelect.value = "Maruti";
        updateModelDropdown();
    }

    function updateModelDropdown() {
        if (!companySelect || !modelSelect) return;
        const brand = companySelect.value;
        const modelsObj = vehicleHierarchy[brand] || {};

        modelSelect.innerHTML = "";
        Object.keys(modelsObj).forEach(m => {
            const opt = document.createElement("option");
            opt.value = m;
            opt.innerText = m;
            modelSelect.appendChild(opt);
        });

        if (modelSelect.options.length > 0) {
            modelSelect.selectedIndex = 0;
        }
        updateVariantDropdown();
    }

    function updateVariantDropdown() {
        if (!companySelect || !modelSelect || !variantSelect) return;
        const brand = companySelect.value;
        const model = modelSelect.value;
        const variants = (vehicleHierarchy[brand] && vehicleHierarchy[brand][model]) || ["Base"];

        variantSelect.innerHTML = "";
        variants.forEach(v => {
            const opt = document.createElement("option");
            opt.value = v;
            opt.innerText = v;
            variantSelect.appendChild(opt);
        });

        if (variantSelect.options.length > 0) {
            variantSelect.selectedIndex = 0;
        }
    }

    if (companySelect && modelSelect && variantSelect) {
        companySelect.addEventListener("change", updateModelDropdown);
        modelSelect.addEventListener("change", updateVariantDropdown);
        initHierarchy();
    }

    // 2. Tab Navigation System
    const navLinks = document.querySelectorAll(".nav-link");
    const tabPanes = document.querySelectorAll(".tab-pane");

    navLinks.forEach(link => {
        link.addEventListener("click", () => {
            const targetId = link.getAttribute("data-tab");

            navLinks.forEach(l => l.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            link.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
            }

            if (targetId === "trend-tab") {
                renderTrendChart("Maruti Swift");
            }
        });
    });

    // 3. Currency & Language Dropdown Handlers
    const currencySelect = document.getElementById("currency-select");
    const langSelect = document.getElementById("lang-select");

    let lastValuationPayload = null;
    let currentPredictionId = null;

    if (currencySelect) {
        currencySelect.addEventListener("change", () => {
            if (lastValuationPayload) {
                triggerValuation(lastValuationPayload);
            }
        });
    }

    if (langSelect) {
        langSelect.addEventListener("change", () => {
            if (lastValuationPayload) {
                triggerValuation(lastValuationPayload);
            }
        });
    }

    // 4. Valuation Form Submission
    const valuationForm = document.getElementById("valuation-form");

    if (valuationForm) {
        valuationForm.addEventListener("submit", (e) => {
            e.preventDefault();

            const formData = new FormData(valuationForm);
            lastValuationPayload = {
                company_name: formData.get("company_name"),
                model_name: formData.get("model_name"),
                variant_name: formData.get("variant_name"),
                manufacture_year: parseInt(formData.get("manufacture_year")),
                km_driven: parseFloat(formData.get("km_driven")),
                city: formData.get("city"),
                fuel_type: formData.get("fuel_type"),
                transmission: formData.get("transmission"),
                description: formData.get("description") || null
            };

            triggerValuation(lastValuationPayload);
        });
    }

    async function triggerValuation(payload) {
        const curr = currencySelect ? currencySelect.value : "INR";
        const lang = langSelect ? langSelect.value : "en";

        try {
            const response = await fetch(`/predict/localized?currency=${curr}&lang=${lang}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Prediction API Error");

            const result = await response.json();
            currentPredictionId = result.prediction_id;

            // Fetch Fraud Audit
            try {
                const fraudRes = await fetch("/fraud/evaluate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (fraudRes.ok) {
                    result.fraud_report = await fraudRes.json();
                }
            } catch (e) {}

            // Fetch Dealer Tiers
            try {
                const dealerRes = await fetch("/dealer/analytics", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (dealerRes.ok) {
                    result.dealer_analytics = await dealerRes.json();
                }
            } catch (e) {}

            displayValuationOutput(result);
        } catch (err) {
            alert("Error calculating valuation. Please verify server connection.");
        }
    }

    function displayValuationOutput(data) {
        document.getElementById("valuation-placeholder").classList.add("hidden");
        document.getElementById("valuation-output").classList.remove("hidden");

        const sym = getCurrencySymbol(data.currency || "INR");

        // Price Hero
        document.getElementById("predicted-price-text").innerText = data.formatted_price || (sym + Math.round(data.predicted_price).toLocaleString());
        document.getElementById("price-low-val").innerText = sym + Math.round(data.price_range_low).toLocaleString();
        document.getElementById("price-high-val").innerText = sym + Math.round(data.price_range_high).toLocaleString();

        if (data.timestamp) {
            document.getElementById("pred-timestamp").innerText = data.timestamp.split("T")[0];
        }

        // Fraud & NLP Badges
        if (data.fraud_report) {
            document.getElementById("fraud-level-text").innerText = `${data.fraud_report.risk_level} RISK (${data.fraud_report.fraud_risk_score})`;
            const badge = document.getElementById("fraud-badge");
            badge.className = `audit-badge ${data.fraud_report.risk_level === 'HIGH' ? 'badge-danger' : 'badge-safe'}`;
        }

        if (data.description_quality_score !== null && data.description_quality_score !== undefined) {
            const score = data.description_quality_score;
            const label = score >= 0.7 ? "High" : (score >= 0.45 ? "Medium" : "Low");
            document.getElementById("nlp-score-text").innerText = `${label} (${score.toFixed(2)})`;
        }

        // SHAP Breakdown Bars
        const shapContainer = document.getElementById("shap-bars-container");
        shapContainer.innerHTML = "";

        if (data.shap_breakdown && data.shap_breakdown.length > 0) {
            data.shap_breakdown.forEach(item => {
                const row = document.createElement("div");
                row.className = "shap-row";

                const isPos = item.impact_inr >= 0;
                const valFormatted = (isPos ? "+" : "-") + sym + Math.abs(Math.round(item.impact_inr)).toLocaleString();
                const cls = isPos ? "shap-val-pos" : "shap-val-neg";

                row.innerHTML = `
                    <span>${item.feature}</span>
                    <span class="${cls}">${valFormatted}</span>
                `;
                shapContainer.appendChild(row);
            });
        }

        // Commercial Dealer Pricing Tiers
        if (data.dealer_analytics && data.dealer_analytics.pricing_tiers) {
            const tiers = data.dealer_analytics.pricing_tiers;
            document.getElementById("tier-tradein-val").innerText = sym + Math.round(tiers.trade_in_wholesale).toLocaleString();
            document.getElementById("tier-private-val").innerText = sym + Math.round(tiers.private_party).toLocaleString();
            document.getElementById("tier-retail-val").innerText = sym + Math.round(tiers.retail_showroom).toLocaleString();
        }
    }

    function getCurrencySymbol(curr) {
        const map = { "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "AED": "AED ", "JPY": "¥" };
        return map[curr] || "₹";
    }

    // 5. PDF Download Handler
    const downloadPdfBtn = document.getElementById("download-pdf-btn");
    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener("click", () => {
            if (!currentPredictionId) return;
            window.open(`/report/${currentPredictionId}`, "_blank");
        });
    }

    // 6. Comparable Listings Handler
    const findCompBtn = document.getElementById("find-comparable-btn");
    if (findCompBtn) {
        findCompBtn.addEventListener("click", async () => {
            if (!lastValuationPayload) return;
            const { company_name, model_name, manufacture_year, city } = lastValuationPayload;

            try {
                const res = await fetch(`/compare?company=${company_name}&model=${model_name}&year=${manufacture_year}&city=${city}`);
                const data = await res.json();

                const box = document.getElementById("comparable-listings-box");
                const list = document.getElementById("comparable-list");
                list.innerHTML = "";

                if (data.listings && data.listings.length > 0) {
                    data.listings.forEach(item => {
                        const div = document.createElement("div");
                        div.className = "comp-item";
                        div.innerHTML = `
                            <span><strong>${item.company_name} ${item.model_name} ${item.variant_name}</strong> (${item.manufacture_year})</span>
                            <span>₹${item.asking_price.toLocaleString()} • ${item.km_driven.toLocaleString()} km</span>
                        `;
                        list.appendChild(div);
                    });
                } else {
                    list.innerHTML = "<p class='text-muted'>No direct comparable matches in database sample.</p>";
                }

                box.classList.remove("hidden");
            } catch (e) {
                alert("Error fetching comparable listings.");
            }
        });
    }

    // 7. Chatbot Form Handler
    const chatForm = document.getElementById("chat-form");
    if (chatForm) {
        chatForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const input = document.getElementById("chat-input");
            const text = input.value.trim();
            if (!text) return;

            appendChatMessage("user", text);
            input.value = "";

            try {
                const res = await fetch("/chat-predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: text })
                });
                const data = await res.json();

                if (data.prediction) {
                    const price = "₹" + Math.round(data.prediction.predicted_price).toLocaleString("en-IN");
                    const low = "₹" + Math.round(data.prediction.price_range_low).toLocaleString("en-IN");
                    const high = "₹" + Math.round(data.prediction.price_range_high).toLocaleString("en-IN");

                    const reply = `I parsed your query as: <b>${data.parsed.company_name} ${data.parsed.model_name} ${data.parsed.variant_name} (${data.parsed.manufacture_year})</b>, ${data.parsed.km_driven.toLocaleString()} km in ${data.parsed.city}.<br><br>` +
                                  `💰 <b>Estimated Valuation: ${price}</b><br>` +
                                  `📊 Confidence Range: ${low} – ${high}`;
                    appendChatMessage("bot", reply);
                } else {
                    appendChatMessage("bot", "Could not parse query. Try entering year, model, and city.");
                }
            } catch (err) {
                appendChatMessage("bot", "Error calculating chat valuation.");
            }
        });
    }

    window.fillChatPrompt = function(promptText) {
        const chatInput = document.getElementById("chat-input");
        if (chatInput) {
            chatInput.value = promptText;
            chatInput.focus();
        }
    };

    function appendChatMessage(sender, htmlContent) {
        const chatBox = document.getElementById("chat-messages");
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-msg ${sender}`;

        const icon = sender === "bot" ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';
        msgDiv.innerHTML = `<div class="msg-avatar">${icon}</div><div class="msg-content">${htmlContent}</div>`;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // 8. PyTorch CNN Image Upload Handler
    const dropZone = document.getElementById("drop-zone");
    const imageUpload = document.getElementById("image-upload");

    if (dropZone && imageUpload) {
        dropZone.addEventListener("click", () => imageUpload.click());

        imageUpload.addEventListener("change", async (e) => {
            if (!e.target.files.length) return;
            const file = e.target.files[0];

            const formData = new FormData();
            formData.append("file", file);

            try {
                const res = await fetch("/upload-image", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();

                const imgResult = document.getElementById("image-result");
                const preview = document.getElementById("uploaded-preview");
                preview.src = URL.createObjectURL(file);

                document.getElementById("tier-badge").innerText = `Tier ${data.predicted_tier}: ${data.condition_label}`;
                document.getElementById("condition-score-val").innerText = `Visual Score: ${data.visual_condition_score.toFixed(2)} / 1.00`;
                document.getElementById("condition-label-text").innerText = `MobileNetV3 CNN defect analysis completed.`;

                imgResult.classList.remove("hidden");
            } catch (err) {
                alert("Error analyzing image.");
            }
        });
    }

    // 9. Chart.js Price Trend Render
    let trendChartInstance = null;

    async function renderTrendChart(variantKey) {
        const ctx = document.getElementById("trendChart");
        if (!ctx) return;

        try {
            const res = await fetch(`/trend/${encodeURIComponent(variantKey)}?months=3`);
            const data = await res.json();

            const labels = ["Base", ...data.forecasts.map(f => f.date)];
            const prices = [data.base_price, ...data.forecasts.map(f => f.predicted_price)];

            if (trendChartInstance) {
                trendChartInstance.destroy();
            }

            const chartCtx = ctx.getContext("2d");
            const gradient = chartCtx.createLinearGradient(0, 0, 0, 300);
            gradient.addColorStop(0, "rgba(0, 242, 254, 0.4)");
            gradient.addColorStop(1, "rgba(0, 242, 254, 0.0)");

            trendChartInstance = new Chart(ctx, {
                type: "line",
                data: {
                    labels: labels,
                    datasets: [{
                        label: `Price Forecast for ${variantKey} (INR)`,
                        data: prices,
                        borderColor: "#00f2fe",
                        borderWidth: 3,
                        backgroundColor: gradient,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 6,
                        pointBackgroundColor: "#00f2fe"
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            grid: { color: "rgba(255, 255, 255, 0.05)" },
                            ticks: { color: "#94a3b8" }
                        },
                        x: {
                            grid: { color: "rgba(255, 255, 255, 0.05)" },
                            ticks: { color: "#94a3b8" }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: "#f8fafc" } }
                    }
                }
            });
        } catch (e) {
            console.error("Error rendering trend chart:", e);
        }
    }

    const fetchTrendBtn = document.getElementById("fetch-trend-btn");
    if (fetchTrendBtn) {
        fetchTrendBtn.addEventListener("click", () => {
            const val = document.getElementById("trend-variant-input").value;
            if (val) renderTrendChart(val);
        });
    }
});
