/* 
   AutoValuate AI — Application Logic & API Integration
   Connects UI components to FastAPI backend endpoints.
*/

document.addEventListener("DOMContentLoaded", () => {
    // 1. Tab Navigation
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(targetId).classList.add("active");

            if (targetId === "trend-tab") {
                renderTrendChart("Maruti Swift");
            }
        });
    });

    // 2. Valuation Form Submission
    const valuationForm = document.getElementById("valuation-form");
    let currentPredictionId = null;

    if (valuationForm) {
        valuationForm.addEventListener("submit", async (e) => {
            e.preventDefault();

            const formData = new FormData(valuationForm);
            const payload = {
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

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) throw new Error("Prediction API Error");

                const result = await response.json();
                currentPredictionId = result.prediction_id;

                displayValuationResult(result);
            } catch (err) {
                alert("Error calculating valuation. Check server status.");
            }
        });
    }

    function displayValuationResult(data) {
        document.getElementById("valuation-placeholder").classList.add("hidden");
        document.getElementById("valuation-output").classList.remove("hidden");

        const formattedPrice = "₹" + Math.round(data.predicted_price).toLocaleString("en-IN");
        const formattedLow = "₹" + Math.round(data.price_range_low).toLocaleString("en-IN");
        const formattedHigh = "₹" + Math.round(data.price_range_high).toLocaleString("en-IN");

        document.getElementById("predicted-price-text").innerText = formattedPrice;
        document.getElementById("price-range-text").innerText = `${formattedLow} – ${formattedHigh}`;

        // SHAP breakdown
        const shapContainer = document.getElementById("shap-bars-container");
        shapContainer.innerHTML = "";

        data.shap_breakdown.forEach(item => {
            const div = document.createElement("div");
            div.className = "shap-item";
            const valFormatted = (item.impact_inr > 0 ? "+₹" : "-₹") + Math.abs(Math.round(item.impact_inr)).toLocaleString("en-IN");
            const cls = item.impact_inr >= 0 ? "impact-pos" : "impact-neg";

            div.innerHTML = `
                <span>${item.feature}</span>
                <span class="${cls}">${valFormatted}</span>
            `;
            shapContainer.appendChild(div);
        });
    }

    // 3. PDF Download Handler
    const downloadPdfBtn = document.getElementById("download-pdf-btn");
    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener("click", () => {
            if (!currentPredictionId) return;
            window.open(`/report/${currentPredictionId}`, "_blank");
        });
    }

    // 4. Comparable Listings Handler
    const findCompBtn = document.getElementById("find-comparable-btn");
    if (findCompBtn) {
        findCompBtn.addEventListener("click", async () => {
            const company = document.getElementById("company_name").value;
            const model = document.getElementById("model_name").value;
            const year = document.getElementById("manufacture_year").value;
            const city = document.getElementById("city").value;

            try {
                const res = await fetch(`/compare?company=${company}&model=${model}&year=${year}&city=${city}`);
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
                            <span>₹${item.asking_price.toLocaleString("en-IN")} • ${item.km_driven.toLocaleString("en-IN")} km</span>
                        `;
                        list.appendChild(div);
                    });
                } else {
                    list.innerHTML = "<p>No direct matches found in current database sample.</p>";
                }

                box.classList.remove("hidden");
            } catch (e) {
                alert("Error fetching comparables.");
            }
        });
    }

    // 5. Chatbot Form Handler
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

                    const reply = `I parsed your vehicle query as: <b>${data.parsed.company_name} ${data.parsed.model_name} ${data.parsed.variant_name} (${data.parsed.manufacture_year})</b>, ${data.parsed.km_driven.toLocaleString()} km in ${data.parsed.city}.<br><br>` +
                                  `💰 <b>Estimated Valuation: ${price}</b><br>` +
                                  `📊 Confidence Range: ${low} – ${high}`;
                    appendChatMessage("bot", reply);
                } else {
                    appendChatMessage("bot", "Could not parse query properly. Please try including year, model, and city.");
                }
            } catch (err) {
                appendChatMessage("bot", "Sorry, an error occurred while calculating valuation.");
            }
        });
    }

    function appendChatMessage(sender, htmlContent) {
        const chatBox = document.getElementById("chat-messages");
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-message`;

        const icon = sender === "bot" ? '<i class="fa-solid fa-robot message-icon"></i>' : '<i class="fa-solid fa-user message-icon"></i>';
        msgDiv.innerHTML = `${icon}<div class="message-bubble">${htmlContent}</div>`;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // 6. PyTorch Image Upload Handler
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
                document.getElementById("condition-label-text").innerText = `Model predictions complete with PyTorch MobileNetV3 CNN.`;

                imgResult.classList.remove("hidden");
            } catch (err) {
                alert("Error processing image.");
            }
        });
    }

    // 7. Chart.js Price Trend Render
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

            trendChartInstance = new Chart(ctx, {
                type: "line",
                data: {
                    labels: labels,
                    datasets: [{
                        label: `Price Forecast for ${variantKey} (INR)`,
                        data: prices,
                        borderColor: "#00f2fe",
                        backgroundColor: "rgba(0, 242, 254, 0.1)",
                        fill: true,
                        tension: 0.3,
                        pointRadius: 6
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
                        legend: { labels: { color: "#f1f5f9" } }
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
