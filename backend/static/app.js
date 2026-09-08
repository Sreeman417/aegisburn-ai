"use strict";

/* ============================================================
   AEGISBURN AI — FRONTEND APPLICATION
   ============================================================ */

const API_BASE = window.location.origin;

const state = {
    components: [],
    selectedComponentId: null,
    selectedResult: null
};


/* ============================================================
   DOM HELPERS
   ============================================================ */

function $(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const element = $(id);

    if (!element) {
        return;
    }

    element.textContent =
        value === undefined || value === null
            ? ""
            : String(value);
}

function setHTML(id, value) {
    const element = $(id);

    if (!element) {
        return;
    }

    element.innerHTML = value;
}

function formatNumber(value, decimals = 2) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function formatValue(value, unit = "") {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return `${formatNumber(number)}${unit ? ` ${unit}` : ""}`;
}

function escapeHTML(value) {
    if (value === undefined || value === null) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


/* ============================================================
   RISK HELPERS
   ============================================================ */

function normalizeRisk(value) {
    if (!value) {
        return "REVIEW";
    }

    const risk = String(value).toUpperCase();

    if (
        risk.includes("HIGH") ||
        risk.includes("CRITICAL")
    ) {
        return "HIGH RISK";
    }

    if (
        risk.includes("NORMAL") ||
        risk.includes("LOW") ||
        risk.includes("SAFE")
    ) {
        return "NORMAL";
    }

    return "REVIEW";
}

function riskClass(risk) {
    const normalized = normalizeRisk(risk);

    if (normalized === "HIGH RISK") {
        return "high-risk";
    }

    if (normalized === "NORMAL") {
        return "normal";
    }

    return "review";
}


/* ============================================================
   API REQUEST
   ============================================================ */

async function apiRequest(path, options = {}) {
    const requestOptions = {
        method: options.method || "GET",
        headers: {
            ...(options.headers || {})
        }
    };

    if (options.body !== undefined) {
        requestOptions.headers["Content-Type"] =
            "application/json";

        requestOptions.body =
            JSON.stringify(options.body);
    }

    const response = await fetch(
        `${API_BASE}${path}`,
        requestOptions
    );

    const contentType =
        response.headers.get("content-type") || "";

    let data;

    if (contentType.includes("application/json")) {
        data = await response.json();
    } else {
        const text = await response.text();

        try {
            data = JSON.parse(text);
        } catch {
            data = {
                detail: text
            };
        }
    }

    if (!response.ok) {
        const message =
            data?.detail ||
            data?.message ||
            `Request failed with status ${response.status}`;

        throw new Error(
            typeof message === "string"
                ? message
                : JSON.stringify(message)
        );
    }

    return data;
}


/* ============================================================
   CONNECTION STATUS
   ============================================================ */

function getConnectionElement() {
    return (
        $("connectionText") ||
        $("connectionStatus") ||
        $("systemStatus")
    );
}

function getConnectionDot() {
    return $("connectionDot") || $("statusDot");
}

function setConnectionState(
    stateName,
    text
) {
    const textElement =
        getConnectionElement();

    const dotElement =
        getConnectionDot();

    if (textElement) {
        textElement.textContent = text;

        textElement.classList.remove(
            "online",
            "offline",
            "connecting"
        );

        textElement.classList.add(
            stateName
        );
    }

    if (dotElement) {
        dotElement.classList.remove(
            "online",
            "offline",
            "connecting"
        );

        dotElement.classList.add(
            stateName
        );
    }
}

async function checkConnection() {
    try {
        const health =
            await apiRequest("/health");

        if (
            health &&
            health.status === "healthy"
        ) {
            setConnectionState(
                "online",
                "System Online"
            );
        } else {
            setConnectionState(
                "online",
                "System Ready"
            );
        }

        return true;

    } catch (error) {
        console.error(
            "Backend connection failed:",
            error
        );

        setConnectionState(
            "offline",
            "Backend Offline"
        );

        return false;
    }
}


/* ============================================================
   RESPONSE NORMALIZATION
   ============================================================ */

function extractComponents(data) {
    if (Array.isArray(data)) {
        return data;
    }

    if (Array.isArray(data?.components)) {
        return data.components;
    }

    if (Array.isArray(data?.data)) {
        return data.data;
    }

    if (Array.isArray(data?.items)) {
        return data.items;
    }

    return [];
}

function extractAnalysis(data) {
    if (!data) {
        return {};
    }

    if (data.result) {
        return data.result;
    }

    if (data.analysis) {
        return data.analysis;
    }

    if (data.data) {
        return data.data;
    }

    return data;
}


/* ============================================================
   LOAD COMPONENTS
   ============================================================ */

async function loadComponents() {
    const response =
        await apiRequest("/components");

    const components =
        extractComponents(response);

    state.components = components;

    populateComponentSelector(
        components
    );

    return components;
}

function getComponentId(component) {
    return (
        component?.component_id ??
        component?.id ??
        component?.ComponentID ??
        ""
    );
}

function getComponentType(component) {
    return (
        component?.component_type ??
        component?.type ??
        ""
    );
}

function getParameterName(component) {
    return (
        component?.parameter_name ??
        component?.parameter ??
        ""
    );
}

function populateComponentSelector(
    components
) {
    const select =
        $("componentSelect") ||
        $("component-selector");

    if (!select) {
        console.warn(
            "Component selector not found."
        );

        return;
    }

    select.innerHTML = "";

    if (!components.length) {
        const option =
            document.createElement("option");

        option.value = "";
        option.textContent =
            "No components available";

        select.appendChild(option);

        return;
    }

    components.forEach(
        (component) => {
            const option =
                document.createElement(
                    "option"
                );

            const componentId =
                getComponentId(component);

            const componentType =
                getComponentType(component);

            const parameter =
                getParameterName(component);

            option.value = componentId;

            option.textContent =
                `${componentId} — ${componentType} — ${parameter}`;

            select.appendChild(option);
        }
    );

    if (
        state.selectedComponentId &&
        components.some(
            (component) =>
                String(
                    getComponentId(component)
                ) ===
                String(
                    state.selectedComponentId
                )
        )
    ) {
        select.value =
            state.selectedComponentId;
    } else {
        state.selectedComponentId =
            select.value;
    }
}


/* ============================================================
   COMPONENT LOOKUP
   ============================================================ */

function findComponent(componentId) {
    return state.components.find(
        (component) =>
            String(
                getComponentId(component)
            ) ===
            String(componentId)
    );
}


/* ============================================================
   ENGINEERING LIMITS
   ============================================================ */

const ENGINEERING_LIMITS = {
    "Logic IC": {
        "Iddq": 50
    },

    "Memory IC": {
        "Standby Current": 80
    },

    "ADC": {
        "Leakage Current": 25
    },

    "Driver IC": {
        "Propagation Delay": 40
    }
};

function getEngineeringLimit(data) {
    const explicitLimit =
        Number(
            data?.engineering_limit
        );

    if (
        Number.isFinite(
            explicitLimit
        ) &&
        explicitLimit > 0
    ) {
        return explicitLimit;
    }

    const componentType =
        data?.component_type ||
        data?.type ||
        "";

    const parameter =
        data?.parameter_name ||
        data?.parameter ||
        "";

    return (
        ENGINEERING_LIMITS[
            componentType
        ]?.[parameter] ??
        null
    );
}

function calculateLimitUtilization(data) {
    const limit =
        getEngineeringLimit(data);

    if (
        !Number.isFinite(limit) ||
        limit <= 0
    ) {
        return null;
    }

    const currentValue =
        Number(
            data?.value_168h ??
            data?.value_96h ??
            data?.value_24h ??
            data?.value_0h
        );

    if (
        !Number.isFinite(currentValue)
    ) {
        return null;
    }

    return (
        currentValue /
        limit *
        100
    );
}


/* ============================================================
   LOAD AND ANALYZE SINGLE COMPONENT
   ============================================================ */

async function loadComponent(
    componentId
) {
    if (!componentId) {
        return;
    }

    state.selectedComponentId =
        componentId;

    setText(
        "componentStatus",
        `${componentId} screening in progress...`
    );

    try {
        /*
         * Use the analysis endpoint first.
         * This is the important endpoint because
         * it runs anomaly detection, prediction,
         * and risk assessment.
         */

        let response;

        try {
            response =
                await apiRequest(
                    `/analyze/${encodeURIComponent(
                        componentId
                    )}`
                );

        } catch (analysisError) {
            console.warn(
                "GET /analyze/{component_id} failed:",
                analysisError
            );

            /*
             * Compatibility fallback.
             */

            try {
                response =
                    await apiRequest(
                        `/components/${encodeURIComponent(
                            componentId
                        )}`
                    );

            } catch (componentError) {
                console.warn(
                    "GET /components/{component_id} failed:",
                    componentError
                );

                response =
                    findComponent(
                        componentId
                    ) || {};
            }
        }

        const result =
            extractAnalysis(response);

        const baseComponent =
            findComponent(
                componentId
            ) || {};

        state.selectedResult = {
            ...baseComponent,
            ...result
        };

        renderAnalysis(
            state.selectedResult
        );

        setText(
            "componentStatus",
            `${componentId} AI screening complete.`
        );

    } catch (error) {
        console.error(
            "Component analysis failed:",
            error
        );

        setText(
            "componentStatus",
            `Screening failed: ${error.message}`
        );

        showError(
            error.message
        );
    }
}


/* ============================================================
   OPTIONAL POST ANALYSIS
   ============================================================ */

async function analyzeSelectedComponent() {
    const componentId =
        state.selectedComponentId;

    if (!componentId) {
        return;
    }

    try {
        const response =
            await apiRequest(
                "/analyze",
                {
                    method: "POST",
                    body: {
                        component_id:
                            componentId
                    }
                }
            );

        const result =
            extractAnalysis(response);

        state.selectedResult = {
            ...(findComponent(
                componentId
            ) || {}),
            ...result
        };

        renderAnalysis(
            state.selectedResult
        );

    } catch (error) {
        console.warn(
            "POST /analyze failed:",
            error
        );

        await loadComponent(
            componentId
        );
    }
}


/* ============================================================
   RENDER COMPLETE ANALYSIS
   ============================================================ */

function renderAnalysis(data) {
    if (!data) {
        return;
    }

    renderOverview(data);
    renderRiskPanel(data);
    renderPrediction(data);
    renderReasons(data);
    renderGraph(data);
    renderEngineeringLimit(data);
}


/* ============================================================
   COMPONENT OVERVIEW
   ============================================================ */

function renderOverview(data) {
    const componentId =
        data.component_id ??
        data.id ??
        "—";

    const componentType =
        data.component_type ??
        data.type ??
        "—";

    const parameter =
        data.parameter_name ??
        data.parameter ??
        "—";

    const lot =
        data.lot_id ??
        data.lot ??
        "—";

    const unit =
        data.unit ??
        "";

    const risk =
        normalizeRisk(
            data.risk_level ??
            data.risk
        );

    const value0 =
        data.value_0h ??
        data["0h"];

    const value24 =
        data.value_24h ??
        data["24h"];

    const overviewBody =
        $("overviewBody") ||
        $("componentOverviewBody");

    if (overviewBody) {
        overviewBody.innerHTML = `
            <tr>
                <td>
                    ${escapeHTML(
                        componentId
                    )}
                </td>

                <td>
                    ${escapeHTML(
                        componentType
                    )}
                </td>

                <td>
                    ${escapeHTML(
                        parameter
                    )}
                </td>

                <td>
                    ${escapeHTML(
                        lot
                    )}
                </td>

                <td>
                    ${escapeHTML(
                        formatValue(
                            value0,
                            unit
                        )
                    )}
                </td>

                <td>
                    ${escapeHTML(
                        formatValue(
                            value24,
                            unit
                        )
                    )}
                </td>

                <td>
                    <span
                        class="risk-badge ${riskClass(
                            risk
                        )}"
                    >
                        ${escapeHTML(
                            risk
                        )}
                    </span>
                </td>
            </tr>
        `;
    }

    setText(
        "detailComponentId",
        componentId
    );

    setText(
        "detailComponentType",
        componentType
    );

    setText(
        "detailParameter",
        parameter
    );

    setText(
        "detailLot",
        lot
    );
}


/* ============================================================
   RISK PANEL
   ============================================================ */

function renderRiskPanel(data) {
    const riskScore =
        data.risk_score ??
        data.riskScore;

    const riskLevel =
        normalizeRisk(
            data.risk_level ??
            data.risk ??
            data.status
        );

    const anomalyScore =
        data.anomaly_score ??
        data.anomaly_index ??
        data.anomalyIndex;

    const anomalyLabel =
        data.anomaly_label ??
        (
            data.anomaly_flag
                ? "ANOMALY"
                : "NORMAL"
        );

    const earlyDrift =
        data.early_drift_index ??
        data.earlyDriftIndex;

    const futureDrift =
        data.future_drift_index ??
        data.futureDriftIndex;

    let limitUtilization =
        data.limit_utilization ??
        data.limitUtilization;

    if (
        !Number.isFinite(
            Number(limitUtilization)
        )
    ) {
        limitUtilization =
            calculateLimitUtilization(
                data
            );
    }

    setText(
        "riskScore",
        Number.isFinite(
            Number(riskScore)
        )
            ? formatNumber(
                riskScore,
                1
            )
            : "—"
    );

    setText(
        "riskLevel",
        riskLevel
    );

    setText(
        "anomalyScore",
        Number.isFinite(
            Number(anomalyScore)
        )
            ? formatNumber(
                anomalyScore,
                2
            )
            : "—"
    );

    setText(
        "anomalyLabel",
        anomalyLabel
    );

    setText(
        "earlyDriftIndex",
        Number.isFinite(
            Number(earlyDrift)
        )
            ? formatNumber(
                earlyDrift,
                1
            )
            : "—"
    );

    setText(
        "futureDriftIndex",
        Number.isFinite(
            Number(futureDrift)
        )
            ? formatNumber(
                futureDrift,
                1
            )
            : "—"
    );

    setText(
        "limitUtilization",
        Number.isFinite(
            Number(limitUtilization)
        )
            ? `${formatNumber(
                limitUtilization,
                2
            )}%`
            : "—"
    );

    const riskElement =
        $("riskLevel");

    if (riskElement) {
        riskElement.className =
            `risk-value ${riskClass(
                riskLevel
            )}`;
    }
}


/* ============================================================
   ENGINEERING LIMIT PANEL
   ============================================================ */

function renderEngineeringLimit(data) {
    const limit =
        getEngineeringLimit(data);

    const unit =
        data?.unit ||
        "";

    const utilization =
        data?.limit_utilization ??
        calculateLimitUtilization(
            data
        );

    const limitElement =
        $("engineeringLimit");

    if (limitElement) {
        if (
            Number.isFinite(
                Number(limit)
            )
        ) {
            limitElement.textContent =
                `${formatNumber(
                    limit,
                    2
                )}${unit ? ` ${unit}` : ""}`;
        } else {
            limitElement.textContent =
                "—";
        }
    }

    const utilizationElement =
        $("limitUtilization");

    if (utilizationElement) {
        if (
            Number.isFinite(
                Number(utilization)
            )
        ) {
            utilizationElement.textContent =
                `${formatNumber(
                    utilization,
                    2
                )}%`;
        } else {
            utilizationElement.textContent =
                "—";
        }
    }

    const progress =
        $("limitProgress") ||
        $("utilizationProgress");

    if (progress) {
        if (
            Number.isFinite(
                Number(utilization)
            )
        ) {
            const percentage =
                Math.max(
                    0,
                    Math.min(
                        100,
                        Number(
                            utilization
                        )
                    )
                );

            if (
                progress.tagName ===
                "PROGRESS"
            ) {
                progress.value =
                    percentage;

                progress.max =
                    100;
            } else {
                progress.style.width =
                    `${percentage}%`;
            }
        } else {
            if (
                progress.tagName ===
                "PROGRESS"
            ) {
                progress.value = 0;
            } else {
                progress.style.width =
                    "0%";
            }
        }
    }
}


/* ============================================================
   PREDICTION
   ============================================================ */

function renderPrediction(data) {
    const unit =
        data.unit ||
        "";

    const predicted168 =
        data.predicted_168h ??
        data.predicted168h;

    const actual168 =
        data.value_168h ??
        data["168h"];

    const slope =
        data.predicted_slope ??
        data.predictedSlope;

    const safetySlope =
        data.safety_slope ??
        data.safetySlope;

    setText(
        "predicted168h",
        formatValue(
            predicted168,
            unit
        )
    );

    setText(
        "actual168h",
        formatValue(
            actual168,
            unit
        )
    );

    setText(
        "predictedSlope",
        Number.isFinite(
            Number(slope)
        )
            ? formatNumber(
                slope,
                6
            )
            : "—"
    );

    setText(
        "safetySlope",
        Number.isFinite(
            Number(safetySlope)
        )
            ? formatNumber(
                safetySlope,
                6
            )
            : "—"
    );
}


/* ============================================================
   REASONS / EXPLAINABILITY
   ============================================================ */

function renderReasons(data) {
    const reasons =
        Array.isArray(
            data.reasons
        )
            ? data.reasons
            : [];

    const container =
        $("reasonsList") ||
        $("riskReasons");

    if (!container) {
        return;
    }

    if (!reasons.length) {
        container.innerHTML = `
            <li>
                No additional risk factors reported.
            </li>
        `;

        return;
    }

    container.innerHTML =
        reasons
            .map(
                (reason) => `
                    <li>
                        ${escapeHTML(
                            reason
                        )}
                    </li>
                `
            )
            .join("");
}


/* ============================================================
   BURN-IN GRAPH
   SVG — NO EXTERNAL DEPENDENCY
   ============================================================ */

function renderGraph(data) {
    const container =
        $("burnInChart") ||
        $("trendChart") ||
        $("measurementChart");

    if (!container) {
        console.warn(
            "Chart container not found."
        );

        return;
    }

    const measurements = [
        {
            hour: 0,
            value: Number(
                data.value_0h
            )
        },

        {
            hour: 24,
            value: Number(
                data.value_24h
            )
        },

        {
            hour: 96,
            value: Number(
                data.value_96h
            )
        },

        {
            hour: 168,
            value: Number(
                data.value_168h
            )
        }
    ].filter(
        (point) =>
            Number.isFinite(
                point.value
            )
    );

    const predictedValue =
        Number(
            data.predicted_168h
        );

    if (!measurements.length) {
        container.innerHTML = `
            <div class="chart-empty">
                No measurement data available.
            </div>
        `;

        return;
    }

    const allValues =
        measurements.map(
            (point) =>
                point.value
        );

    if (
        Number.isFinite(
            predictedValue
        )
    ) {
        allValues.push(
            predictedValue
        );
    }

    const width = 1000;
    const height = 420;

    const padding = {
        top: 45,
        right: 40,
        bottom: 65,
        left: 75
    };

    const chartWidth =
        width -
        padding.left -
        padding.right;

    const chartHeight =
        height -
        padding.top -
        padding.bottom;

    let minValue =
        Math.min(
            ...allValues
        );

    let maxValue =
        Math.max(
            ...allValues
        );

    const range =
        maxValue -
        minValue;

    if (range === 0) {
        minValue -= 1;
        maxValue += 1;
    } else {
        minValue -=
            range * 0.12;

        maxValue +=
            range * 0.12;
    }

    const xScale = (hour) =>
        padding.left +
        (hour / 168) *
            chartWidth;

    const yScale = (value) =>
        padding.top +
        (
            (maxValue - value) /
            (maxValue - minValue)
        ) *
            chartHeight;

    const linePoints =
        measurements
            .map(
                (point) =>
                    `${xScale(
                        point.hour
                    )},${yScale(
                        point.value
                    )}`
            )
            .join(" ");

    const areaPoints = [
        `${xScale(
            measurements[0].hour
        )},${padding.top + chartHeight}`,

        ...measurements.map(
            (point) =>
                `${xScale(
                    point.hour
                )},${yScale(
                    point.value
                )}`
        ),

        `${xScale(
            measurements[
                measurements.length - 1
            ].hour
        )},${padding.top + chartHeight}`
    ].join(" ");

    const predictedStart =
        measurements.find(
            (point) =>
                point.hour === 24
        ) ||
        measurements[0];

    let predictionLine = "";

    if (
        predictedStart &&
        Number.isFinite(
            predictedValue
        )
    ) {
        predictionLine = `
            <line
                x1="${xScale(
                    predictedStart.hour
                )}"
                y1="${yScale(
                    predictedStart.value
                )}"
                x2="${xScale(168)}"
                y2="${yScale(
                    predictedValue
                )}"
                class="prediction-line"
            />

            <circle
                cx="${xScale(168)}"
                cy="${yScale(
                    predictedValue
                )}"
                r="6"
                class="prediction-point"
            />
        `;
    }

    const horizontalGridCount = 6;

    let gridLines = "";
    let yLabels = "";

    for (
        let i = 0;
        i <= horizontalGridCount;
        i++
    ) {
        const ratio =
            i /
            horizontalGridCount;

        const y =
            padding.top +
            ratio *
                chartHeight;

        const value =
            maxValue -
            ratio *
                (
                    maxValue -
                    minValue
                );

        gridLines += `
            <line
                x1="${padding.left}"
                y1="${y}"
                x2="${
                    width -
                    padding.right
                }"
                y2="${y}"
                class="chart-grid"
            />
        `;

        yLabels += `
            <text
                x="${
                    padding.left -
                    15
                }"
                y="${y + 5}"
                text-anchor="end"
                class="chart-axis-label"
            >
                ${formatNumber(
                    value,
                    1
                )}
            </text>
        `;
    }

    const xTicks = [
        0,
        24,
        48,
        72,
        96,
        120,
        144,
        168
    ];

    let xLabels = "";

    xTicks.forEach(
        (hour) => {
            const x =
                xScale(hour);

            xLabels += `
                <line
                    x1="${x}"
                    y1="${padding.top}"
                    x2="${x}"
                    y2="${
                        padding.top +
                        chartHeight
                    }"
                    class="chart-grid vertical"
                />

                <text
                    x="${x}"
                    y="${
                        height -
                        30
                    }"
                    text-anchor="middle"
                    class="chart-axis-label"
                >
                    ${hour}h
                </text>
            `;
        }
    );

    const circles =
        measurements
            .map(
                (point) => `
                    <circle
                        cx="${xScale(
                            point.hour
                        )}"
                        cy="${yScale(
                            point.value
                        )}"
                        r="6"
                        class="measurement-point"
                    >
                        <title>
                            ${point.hour}h:
                            ${formatValue(
                                point.value,
                                data.unit ||
                                    ""
                            )}
                        </title>
                    </circle>
                `
            )
            .join("");

    container.innerHTML = `
        <svg
            class="burnin-svg"
            viewBox="0 0 ${width} ${height}"
            role="img"
            aria-label="Burn-in measurement and AI prediction chart"
        >

            ${gridLines}

            ${xLabels}

            ${yLabels}

            <polygon
                points="${areaPoints}"
                class="measurement-area"
            />

            <polyline
                points="${linePoints}"
                class="measurement-line"
            />

            ${predictionLine}

            ${circles}

            <text
                x="${width / 2}"
                y="${height - 5}"
                text-anchor="middle"
                class="chart-axis-title"
            >
                Burn-In Time (Hours)
            </text>

            <text
                x="22"
                y="${height / 2}"
                text-anchor="middle"
                transform="rotate(-90 22 ${
                    height / 2
                })"
                class="chart-axis-title"
            >
                ${escapeHTML(
                    data.parameter_name ||
                    data.parameter ||
                    "Measured Value"
                )}
            </text>
        </svg>

        <div class="chart-legend">

            <div class="legend-item">
                <span
                    class="legend-line measured"
                ></span>

                Measured Value
            </div>

            <div class="legend-item">
                <span
                    class="legend-line predicted"
                ></span>

                AI Predicted Trend
            </div>

        </div>
    `;
}


/* ============================================================
   ERROR DISPLAY
   ============================================================ */

function showError(message) {
    const container =
        $("errorMessage") ||
        $("componentStatus");

    if (!container) {
        return;
    }

    container.textContent =
        `Error: ${message}`;
}


/* ============================================================
   CSV UPLOAD
   ============================================================ */

async function uploadCSV(file) {
    if (!file) {
        return;
    }

    const uploadStatus =
        $("uploadStatus") ||
        $("uploadMessage");

    if (uploadStatus) {
        uploadStatus.textContent =
            "Uploading CSV...";
    }

    try {
        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );

        const response =
            await fetch(
                `${API_BASE}/upload-csv`,
                {
                    method: "POST",
                    body: formData
                }
            );

        const contentType =
            response.headers.get(
                "content-type"
            ) || "";

        let data;

        if (
            contentType.includes(
                "application/json"
            )
        ) {
            data =
                await response.json();
        } else {
            data = {
                detail:
                    await response.text()
            };
        }

        if (!response.ok) {
            throw new Error(
                data?.detail ||
                data?.message ||
                "CSV upload failed."
            );
        }

        if (uploadStatus) {
            uploadStatus.textContent =
                data?.message ||
                "CSV uploaded successfully.";
        }

        state.selectedComponentId =
            null;

        await refreshDashboard();

    } catch (error) {
        console.error(
            "CSV upload failed:",
            error
        );

        if (uploadStatus) {
            uploadStatus.textContent =
                `Upload failed: ${error.message}`;
        }

        showError(
            error.message
        );
    }
}


/* ============================================================
   EVENT LISTENERS
   ============================================================ */

function registerEvents() {
    const select =
        $("componentSelect") ||
        $("component-selector");

    if (select) {
        select.addEventListener(
            "change",
            async (event) => {
                const componentId =
                    event.target.value;

                if (!componentId) {
                    return;
                }

                await loadComponent(
                    componentId
                );
            }
        );
    }

    const refreshButton =
        $("refreshButton") ||
        $("refresh");

    if (refreshButton) {
        refreshButton.addEventListener(
            "click",
            refreshDashboard
        );
    }

    const analyzeButton =
        $("analyzeButton");

    if (analyzeButton) {
        analyzeButton.addEventListener(
            "click",
            analyzeSelectedComponent
        );
    }

    /*
     * Support multiple possible upload
     * input IDs so the frontend remains
     * compatible with the current HTML.
     */

    const fileInput =
        $("csvFile") ||
        $("csvInput") ||
        $("fileInput") ||
        $("uploadFile");

    const uploadButton =
        $("uploadButton") ||
        $("uploadCsvButton") ||
        $("uploadCSVButton");

    if (fileInput) {
        fileInput.addEventListener(
            "change",
            () => {
                const file =
                    fileInput.files?.[0];

                if (file) {
                    const fileName =
                        $("selectedFileName");

                    if (fileName) {
                        fileName.textContent =
                            file.name;
                    }
                }
            }
        );
    }

    if (
        uploadButton &&
        fileInput
    ) {
        uploadButton.addEventListener(
            "click",
            async () => {
                const file =
                    fileInput.files?.[0];

                if (!file) {
                    showError(
                        "Please select a CSV file first."
                    );

                    return;
                }

                await uploadCSV(
                    file
                );
            }
        );
    }
}


/* ============================================================
   REFRESH DASHBOARD
   ============================================================ */

async function refreshDashboard() {
    const refreshButton =
        $("refreshButton") ||
        $("refresh");

    if (refreshButton) {
        refreshButton.disabled =
            true;

        refreshButton.textContent =
            "Refreshing...";
    }

    try {
        const online =
            await checkConnection();

        if (!online) {
            throw new Error(
                "Cannot connect to AegisBurn backend."
            );
        }

        const components =
            await loadComponents();

        const select =
            $("componentSelect") ||
            $("component-selector");

        if (
            components.length > 0 &&
            select
        ) {
            const componentId =
                state.selectedComponentId ||
                select.value;

            if (componentId) {
                select.value =
                    componentId;

                await loadComponent(
                    componentId
                );
            }
        }

    } catch (error) {
        console.error(
            "Dashboard refresh failed:",
            error
        );

        showError(
            error.message
        );

    } finally {
        if (refreshButton) {
            refreshButton.disabled =
                false;

            refreshButton.textContent =
                "Refresh";
        }
    }
}


/* ============================================================
   APPLICATION STARTUP
   ============================================================ */

async function initializeApp() {
    console.log(
        "AEGISBURN AI FRONTEND STARTING..."
    );

    setConnectionState(
        "connecting",
        "Connecting..."
    );

    registerEvents();

    await refreshDashboard();

    console.log(
        "AEGISBURN AI FRONTEND READY"
    );
}


/* ============================================================
   START WHEN PAGE IS READY
   ============================================================ */

if (
    document.readyState ===
    "loading"
) {
    document.addEventListener(
        "DOMContentLoaded",
        initializeApp
    );
} else {
    initializeApp();
}