/* =========================================================
   AEGISBURN AI
   COMPLETE MATCHING FRONTEND SCRIPT

   File:
   backend/static/app.js
   ========================================================= */


/* =========================================================
   CONFIG
   ========================================================= */

const API_BASE = window.location.origin;


/* =========================================================
   DOM HELPERS
   ========================================================= */

function $(id) {
    return document.getElementById(id);
}


function setText(id, value) {
    const element = $(id);

    if (!element) {
        return;
    }

    if (
        value === undefined ||
        value === null ||
        value === ""
    ) {
        element.textContent = "—";
        return;
    }

    element.textContent = value;
}


function numberValue(value, fallback = 0) {
    const n = Number(value);

    return Number.isFinite(n)
        ? n
        : fallback;
}


function formatNumber(value, decimals = 3) {
    if (
        value === undefined ||
        value === null ||
        value === ""
    ) {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toFixed(decimals);
}


function formatInteger(value) {
    if (
        value === undefined ||
        value === null ||
        value === ""
    ) {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return Math.round(number).toLocaleString();
}


function formatPercent(value, decimals = 1) {
    if (
        value === undefined ||
        value === null ||
        value === ""
    ) {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return `${number.toFixed(decimals)}%`;
}


function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


/* =========================================================
   CONNECTION STATUS
   ========================================================= */

function setConnectionStatus(status, message) {
    const element = $("connectionStatus");

    if (!element) {
        return;
    }

    element.className = "system-status";

    const normalized = String(status || "")
        .toLowerCase()
        .trim();

    if (
        normalized === "online" ||
        normalized === "healthy" ||
        normalized === "connected"
    ) {
        element.classList.add("online");
    } else if (
        normalized === "offline" ||
        normalized === "error" ||
        normalized === "failed"
    ) {
        element.classList.add("offline");
    } else {
        element.classList.add("connecting");
    }

    element.textContent =
        message ||
        (
            normalized === "online" ||
            normalized === "healthy" ||
            normalized === "connected"
                ? "System Online"
                : normalized === "offline" ||
                  normalized === "error" ||
                  normalized === "failed"
                    ? "Connection Failed"
                    : "Connecting..."
        );
}


/* =========================================================
   API HELPER
   ========================================================= */

async function apiRequest(path, options = {}) {
    const url = `${API_BASE}${path}`;

    const response = await fetch(url, {
        method: options.method || "GET",
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {})
        },
        body: options.body
            ? JSON.stringify(options.body)
            : undefined
    });

    let data = null;

    try {
        data = await response.json();
    } catch (error) {
        data = null;
    }

    if (!response.ok) {
        let message = `Request failed (${response.status})`;

        if (data) {
            if (typeof data.detail === "string") {
                message = data.detail;
            } else if (typeof data.message === "string") {
                message = data.message;
            } else if (typeof data.error === "string") {
                message = data.error;
            }
        }

        throw new Error(message);
    }

    return data;
}


/* =========================================================
   HEALTH CHECK
   ========================================================= */

async function checkHealth() {
    setConnectionStatus(
        "connecting",
        "Connecting..."
    );

    try {
        const data = await apiRequest("/health");

        const status = String(
            data?.status || "healthy"
        ).toLowerCase();

        if (
            status === "healthy" ||
            status === "running" ||
            status === "ok"
        ) {
            setConnectionStatus(
                "online",
                "System Online"
            );
        } else {
            setConnectionStatus(
                "online",
                "Backend Connected"
            );
        }

        return true;

    } catch (error) {
        console.error(
            "Health check failed:",
            error
        );

        setConnectionStatus(
            "offline",
            "Backend Offline"
        );

        return false;
    }
}


/* =========================================================
   NORMALIZE COMPONENT LIST
   ========================================================= */

function normalizeComponents(data) {

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


/* =========================================================
   GET COMPONENT ID
   ========================================================= */

function getComponentId(component) {
    if (
        typeof component === "string" ||
        typeof component === "number"
    ) {
        return String(component);
    }

    if (!component) {
        return "";
    }

    return String(
        component.component_id ??
        component.id ??
        component.componentId ??
        ""
    );
}


/* =========================================================
   COMPONENT LABEL
   ========================================================= */

function getComponentLabel(component) {

    if (
        typeof component === "string" ||
        typeof component === "number"
    ) {
        return String(component);
    }

    if (!component) {
        return "Unknown Component";
    }

    const id = getComponentId(component);

    const type =
        component.component_type ??
        component.type ??
        "";

    const parameter =
        component.parameter_name ??
        component.parameter ??
        "";

    const parts = [
        id,
        type,
        parameter
    ].filter(Boolean);

    return parts.length > 0
        ? parts.join(" — ")
        : "Unknown Component";
}


/* =========================================================
   LOAD COMPONENTS
   ========================================================= */

async function loadComponents() {

    const select = $("componentSelect");

    if (!select) {
        console.error(
            "componentSelect element not found."
        );
        return [];
    }

    select.disabled = true;

    select.innerHTML = `
        <option value="">
            Loading components...
        </option>
    `;

    try {
        const data = await apiRequest("/components");

        const components =
            normalizeComponents(data);

        select.innerHTML = `
            <option value="">
                Select a component...
            </option>
        `;

        if (components.length === 0) {

            select.innerHTML = `
                <option value="">
                    No components found
                </option>
            `;

            setText(
                "componentStatus",
                "No components were returned by the backend."
            );

            return [];
        }

        components.forEach((component) => {

            const id =
                getComponentId(component);

            if (!id) {
                return;
            }

            const option =
                document.createElement("option");

            option.value = id;

            option.textContent =
                getComponentLabel(component);

            select.appendChild(option);
        });

        select.disabled = false;

        setText(
            "componentStatus",
            `${components.length.toLocaleString()} components available for screening.`
        );

        return components;

    } catch (error) {

        console.error(
            "Failed to load components:",
            error
        );

        select.innerHTML = `
            <option value="">
                Failed to load components
            </option>
        `;

        setText(
            "componentStatus",
            `Failed to load components: ${error.message}`
        );

        return [];
    }
}


/* =========================================================
   SUMMARY COUNTS
   ========================================================= */

function normalizeRisk(value) {
    return String(value || "")
        .trim()
        .toUpperCase()
        .replace(/[_-]/g, " ");
}


function extractSummaryCounts(data) {

    const source =
        data?.summary ||
        data?.counts ||
        data ||
        {};

    const normal =
        source.normal ??
        source.NORMAL ??
        source.normal_count ??
        source.normalCount;

    const review =
        source.review ??
        source.REVIEW ??
        source.review_count ??
        source.reviewCount;

    const high =
        source.high_risk ??
        source.highRisk ??
        source.high ??
        source.HIGH_RISK ??
        source.high_risk_count ??
        source.highRiskCount;

    const total =
        source.total ??
        source.TOTAL ??
        source.total_count ??
        source.totalCount;

    return {
        normal,
        review,
        high,
        total
    };
}


async function loadSummaryCounts(components) {

    let normal = 0;
    let review = 0;
    let high = 0;

    if (!Array.isArray(components)) {
        components = [];
    }

    /*
       First try metadata endpoint.

       If the backend metadata does not contain summary
       information, use the component count and leave
       risk buckets as unavailable instead of crashing.
    */

    try {

        const metadata =
            await apiRequest("/metadata");

        const counts =
            extractSummaryCounts(metadata);

        if (counts.normal !== undefined) {
            normal =
                numberValue(counts.normal);
        }

        if (counts.review !== undefined) {
            review =
                numberValue(counts.review);
        }

        if (counts.high !== undefined) {
            high =
                numberValue(counts.high);
        }

        setText(
            "normalCount",
            counts.normal !== undefined
                ? formatInteger(normal)
                : "—"
        );

        setText(
            "reviewCount",
            counts.review !== undefined
                ? formatInteger(review)
                : "—"
        );

        setText(
            "highRiskCount",
            counts.high !== undefined
                ? formatInteger(high)
                : "—"
        );

        setText(
            "totalCount",
            counts.total !== undefined
                ? formatInteger(counts.total)
                : formatInteger(components.length)
        );

        return;

    } catch (error) {

        console.warn(
            "Metadata summary unavailable:",
            error
        );
    }

    /*
       Fallback.
    */

    setText(
        "normalCount",
        "—"
    );

    setText(
        "reviewCount",
        "—"
    );

    setText(
        "highRiskCount",
        "—"
    );

    setText(
        "totalCount",
        formatInteger(components.length)
    );
}


/* =========================================================
   ANALYZE COMPONENT
   ========================================================= */

async function analyzeComponent(componentId) {

    if (!componentId) {
        return null;
    }

    /*
       Backend route may expect:
       POST /analyze
       {
           "component_id": "CMP-000001"
       }
    */

    return await apiRequest(
        "/analyze",
        {
            method: "POST",

            body: {
                component_id: componentId
            }
        }
    );
}


/* =========================================================
   NORMALIZE ANALYSIS RESPONSE
   ========================================================= */

function normalizeAnalysis(data) {

    if (!data) {
        return {};
    }

    /*
       Supports either:

       {
           component_id: ...
           risk_score: ...
       }

       or:

       {
           result: {
               ...
           }
       }

       or:

       {
           analysis: {
               ...
           }
       }
    */

    if (
        data.result &&
        typeof data.result === "object"
    ) {
        return {
            ...data,
            ...data.result
        };
    }

    if (
        data.analysis &&
        typeof data.analysis === "object"
    ) {
        return {
            ...data,
            ...data.analysis
        };
    }

    return data;
}


/* =========================================================
   OVERVIEW TABLE
   ========================================================= */

function updateOverview(data) {

    const body =
        $("overviewBody");

    if (!body) {
        return;
    }

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

    const value0 =
        data.value_0h ??
        data.value0h ??
        data["0h"];

    const value24 =
        data.value_24h ??
        data.value24h ??
        data["24h"];

    const unit =
        data.unit ??
        "";

    const risk =
        data.risk_level ??
        data.risk ??
        "—";

    const normalizedRisk =
        normalizeRisk(risk);

    let riskClass =
        "risk-review";

    if (
        normalizedRisk.includes("NORMAL") ||
        normalizedRisk.includes("LOW")
    ) {
        riskClass = "risk-normal";
    }

    if (
        normalizedRisk.includes("HIGH")
    ) {
        riskClass = "risk-high";
    }

    body.innerHTML = `
        <tr>
            <td>${escapeHtml(componentId)}</td>

            <td>${escapeHtml(componentType)}</td>

            <td>${escapeHtml(parameter)}</td>

            <td>${escapeHtml(lot)}</td>

            <td>
                ${escapeHtml(formatNumber(value0))}
                ${escapeHtml(unit)}
            </td>

            <td>
                ${escapeHtml(formatNumber(value24))}
                ${escapeHtml(unit)}
            </td>

            <td class="${riskClass}">
                ${escapeHtml(risk)}
            </td>
        </tr>
    `;
}


/* =========================================================
   ANOMALY RESULT
   ========================================================= */

function updateAnomaly(data) {

    const label =
        String(
            data.anomaly_label ??
            data.anomaly ??
            (
                Number(data.anomaly_flag) === 1
                    ? "ANOMALY"
                    : Number(data.anomaly_flag) === 0
                        ? "NORMAL"
                        : "—"
            )
        );

    const element =
        $("anomalyLabel");

    if (element) {

        element.textContent =
            label;

        element.className =
            "result-badge";

        const normalized =
            normalizeRisk(label);

        if (
            normalized.includes("NORMAL")
        ) {
            element.classList.add(
                "badge-normal"
            );
        } else if (
            normalized.includes("HIGH") ||
            normalized.includes("ANOMAL")
        ) {
            element.classList.add(
                "badge-high"
            );
        } else {
            element.classList.add(
                "badge-review"
            );
        }
    }

    setText(
        "anomalyIndex",
        formatNumber(
            data.anomaly_index ??
            data.anomaly_score ??
            data.anomalyIndex,
            2
        )
    );

    setText(
        "anomalyScore",
        formatNumber(
            data.anomaly_score ??
            data.anomaly_raw_score ??
            data.anomalyScore,
            3
        )
    );
}


/* =========================================================
   PREDICTION RESULT
   ========================================================= */

function updatePrediction(data) {

    const unit =
        data.unit ??
        "";

    const predicted =
        data.predicted_168h ??
        data.predicted168h ??
        data.prediction ??
        data.predicted_value;

    const actual =
        data.value_168h ??
        data.actual_168h ??
        data.actual168h;

    const predictedSlope =
        data.predicted_slope ??
        data.predictedSlope;

    const safetySlope =
        data.safety_slope ??
        data.safetySlope;

    setText(
        "predicted168h",
        predicted !== undefined &&
        predicted !== null
            ? `${formatNumber(predicted)} ${unit}`.trim()
            : "—"
    );

    setText(
        "actual168h",
        actual !== undefined &&
        actual !== null
            ? `${formatNumber(actual)} ${unit}`.trim()
            : "—"
    );

    setText(
        "predictedSlope",
        formatNumber(
            predictedSlope,
            6
        )
    );

    setText(
        "safetySlope",
        formatNumber(
            safetySlope,
            6
        )
    );
}


/* =========================================================
   RISK RESULT
   ========================================================= */

function updateRisk(data) {

    const risk =
        data.risk_level ??
        data.risk ??
        "—";

    const normalized =
        normalizeRisk(risk);

    const element =
        $("riskLevel");

    if (element) {

        element.textContent =
            risk;

        element.className =
            "risk-value";

        if (
            normalized.includes("NORMAL") ||
            normalized.includes("LOW")
        ) {
            element.classList.add(
                "normal"
            );
        } else if (
            normalized.includes("HIGH")
        ) {
            element.classList.add(
                "high"
            );
        } else {
            element.classList.add(
                "review"
            );
        }
    }

    setText(
        "riskScore",
        formatNumber(
            data.risk_score ??
            data.riskScore,
            1
        )
    );

    setText(
        "earlyDriftIndex",
        formatNumber(
            data.early_drift_index ??
            data.earlyDriftIndex,
            1
        )
    );

    setText(
        "futureDriftIndex",
        formatNumber(
            data.future_drift_index ??
            data.futureDriftIndex,
            1
        )
    );
}


/* =========================================================
   ENGINEERING LIMIT
   ========================================================= */

function updateEngineeringLimit(data) {

    setText(
        "limitUtilization",
        formatPercent(
            data.limit_utilization ??
            data.limitUtilization,
            2
        )
    );

    const limit =
        data.engineering_limit ??
        data.engineeringLimit;

    const unit =
        data.unit ??
        "";

    setText(
        "engineeringLimit",
        limit !== undefined &&
        limit !== null
            ? `${formatNumber(limit)} ${unit}`.trim()
            : "—"
    );
}


/* =========================================================
   MEASUREMENTS
   ========================================================= */

function updateMeasurements(data) {

    const unit =
        data.unit ??
        "";

    const value0 =
        data.value_0h ??
        data.value0h;

    const value24 =
        data.value_24h ??
        data.value24h;

    const value96 =
        data.value_96h ??
        data.value96h;

    const actual168 =
        data.value_168h ??
        data.actual_168h ??
        data.actual168h;

    const predicted168 =
        data.predicted_168h ??
        data.predicted168h;

    const display = (value) => {

        if (
            value === undefined ||
            value === null
        ) {
            return "—";
        }

        return `${formatNumber(value)} ${unit}`.trim();
    };

    setText(
        "value0h",
        display(value0)
    );

    setText(
        "value24h",
        display(value24)
    );

    setText(
        "value96h",
        display(value96)
    );

    setText(
        "actualMeasurement168h",
        display(actual168)
    );

    setText(
        "measurementPrediction",
        display(predicted168)
    );
}


/* =========================================================
   EXPLAINABILITY REASONS
   ========================================================= */

function updateReasons(data) {

    const list =
        $("reasonsList");

    if (!list) {
        return;
    }

    let reasons =
        data.reasons ??
        data.explanations ??
        data.reasoning ??
        [];

    if (!Array.isArray(reasons)) {

        if (typeof reasons === "string") {
            reasons = [reasons];
        } else {
            reasons = [];
        }
    }

    if (reasons.length === 0) {

        reasons = [
            "No additional risk explanations were returned for this component."
        ];
    }

    list.innerHTML =
        reasons
            .map((reason) => `
                <li>
                    ${escapeHtml(reason)}
                </li>
            `)
            .join("");
}


/* =========================================================
   SVG BURN-IN GRAPH
   ========================================================= */

function drawBurnInChart(data) {

    const container =
        $("burnInChart");

    if (!container) {
        return;
    }

    const unit =
        data.unit ??
        "";

    const points = [
        {
            hour: 0,
            label: "0h",
            value: numberValue(
                data.value_0h ??
                data.value0h,
                NaN
            ),
            type: "actual"
        },

        {
            hour: 24,
            label: "24h",
            value: numberValue(
                data.value_24h ??
                data.value24h,
                NaN
            ),
            type: "actual"
        },

        {
            hour: 96,
            label: "96h",
            value: numberValue(
                data.value_96h ??
                data.value96h,
                NaN
            ),
            type: "actual"
        },

        {
            hour: 168,
            label: "168h Actual",
            value: numberValue(
                data.value_168h ??
                data.actual_168h ??
                data.actual168h,
                NaN
            ),
            type: "actual"
        },

        {
            hour: 168,
            label: "168h Predicted",
            value: numberValue(
                data.predicted_168h ??
                data.predicted168h,
                NaN
            ),
            type: "predicted"
        }
    ].filter(
        (point) => Number.isFinite(point.value)
    );

    if (points.length === 0) {

        container.innerHTML = `
            <div class="chart-empty">
                No measurement data available.
            </div>
        `;

        return;
    }


    /*
       SVG dimensions.
    */

    const width = 1000;
    const height = 430;

    const padding = {
        top: 45,
        right: 55,
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


    /*
       X axis uses actual burn-in hours.
    */

    const xScale = (hour) => {

        return (
            padding.left +
            (
                hour / 168
            ) * chartWidth
        );
    };


    /*
       Y axis.
    */

    const values =
        points.map(
            (point) => point.value
        );

    let minValue =
        Math.min(...values);

    let maxValue =
        Math.max(...values);

    const range =
        maxValue - minValue;

    if (range === 0) {

        minValue -= 1;
        maxValue += 1;

    } else {

        minValue -= range * 0.15;
        maxValue += range * 0.15;
    }

    const yScale = (value) => {

        return (
            padding.top +
            (
                (
                    maxValue - value
                ) /
                (
                    maxValue - minValue
                )
            ) * chartHeight
        );
    };


    /*
       Actual points.
    */

    const actualPoints =
        points
            .filter(
                (point) =>
                    point.type === "actual"
            )
            .sort(
                (a, b) =>
                    a.hour - b.hour
            );


    const predictedPoint =
        points.find(
            (point) =>
                point.type === "predicted"
        );


    const actualPath =
        actualPoints
            .map(
                (point, index) => {

                    const x =
                        xScale(point.hour);

                    const y =
                        yScale(point.value);

                    return `
                        ${index === 0 ? "M" : "L"}
                        ${x}
                        ${y}
                    `;
                }
            )
            .join(" ");


    /*
       Prediction line:
       from last actual point
       to predicted 168h.
    */

    let predictionPath = "";

    if (
        actualPoints.length > 0 &&
        predictedPoint
    ) {

        const lastActual =
            actualPoints[
                actualPoints.length - 1
            ];

        predictionPath = `
            M ${xScale(lastActual.hour)}
              ${yScale(lastActual.value)}

            L ${xScale(predictedPoint.hour)}
              ${yScale(predictedPoint.value)}
        `;
    }


    /*
       Grid lines.
    */

    const gridLines = [];

    const yTicks = 5;

    for (
        let i = 0;
        i <= yTicks;
        i++
    ) {

        const ratio =
            i / yTicks;

        const value =
            maxValue -
            (
                ratio *
                (
                    maxValue -
                    minValue
                )
            );

        const y =
            padding.top +
            ratio * chartHeight;

        gridLines.push(`
            <line
                x1="${padding.left}"
                y1="${y}"
                x2="${width - padding.right}"
                y2="${y}"
                stroke="#26384a"
                stroke-width="1"
            />

            <text
                x="${padding.left - 12}"
                y="${y + 4}"
                text-anchor="end"
                fill="#748092"
                font-size="12"
            >
                ${formatNumber(value, 2)}
            </text>
        `);
    }


    /*
       X ticks.
    */

    const xTicks =
        [0, 24, 96, 168]
            .map((hour) => {

                const x =
                    xScale(hour);

                return `
                    <line
                        x1="${x}"
                        y1="${padding.top}"
                        x2="${x}"
                        y2="${height - padding.bottom}"
                        stroke="#1c2a38"
                        stroke-width="1"
                    />

                    <text
                        x="${x}"
                        y="${height - padding.bottom + 28}"
                        text-anchor="middle"
                        fill="#748092"
                        font-size="12"
                    >
                        ${hour}h
                    </text>
                `;
            })
            .join("");


    /*
       Actual point circles.
    */

    const actualCircles =
        actualPoints
            .map((point) => {

                const x =
                    xScale(point.hour);

                const y =
                    yScale(point.value);

                return `
                    <circle
                        cx="${x}"
                        cy="${y}"
                        r="6"
                        fill="#64d6b4"
                        stroke="#0c141d"
                        stroke-width="3"
                    />

                    <text
                        x="${x}"
                        y="${y - 14}"
                        text-anchor="middle"
                        fill="#aab4c2"
                        font-size="11"
                    >
                        ${formatNumber(point.value, 2)}
                    </text>
                `;
            })
            .join("");


    /*
       Predicted point.
    */

    let predictedCircle = "";

    if (predictedPoint) {

        const x =
            xScale(predictedPoint.hour);

        const y =
            yScale(predictedPoint.value);

        predictedCircle = `
            <circle
                cx="${x}"
                cy="${y}"
                r="7"
                fill="#77adf2"
                stroke="#0c141d"
                stroke-width="3"
            />

            <text
                x="${x - 10}"
                y="${y - 14}"
                text-anchor="end"
                fill="#aab4c2"
                font-size="11"
            >
                Pred: ${formatNumber(predictedPoint.value, 2)}
            </text>
        `;
    }


    /*
       SVG.
    */

    container.innerHTML = `
        <svg
            viewBox="0 0 ${width} ${height}"
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label="Burn-in measurement trend"
        >

            <text
                x="${padding.left}"
                y="24"
                fill="#aab4c2"
                font-size="14"
                font-weight="600"
            >
                Burn-In Measurement Trend (${escapeHtml(unit)})
            </text>


            ${gridLines.join("")}


            ${xTicks}


            <!-- Actual measurement line -->

            <path
                d="${actualPath}"
                fill="none"
                stroke="#64d6b4"
                stroke-width="3"
                stroke-linejoin="round"
                stroke-linecap="round"
            />


            <!-- Prediction line -->

            ${
                predictionPath
                    ? `
                        <path
                            d="${predictionPath}"
                            fill="none"
                            stroke="#77adf2"
                            stroke-width="3"
                            stroke-dasharray="8 7"
                            stroke-linejoin="round"
                            stroke-linecap="round"
                        />
                    `
                    : ""
            }


            ${actualCircles}


            ${predictedCircle}


            <!-- Legend -->

            <line
                x1="${width - 245}"
                y1="26"
                x2="${width - 215}"
                y2="26"
                stroke="#64d6b4"
                stroke-width="3"
            />

            <text
                x="${width - 207}"
                y="30"
                fill="#aab4c2"
                font-size="11"
            >
                Actual
            </text>


            <line
                x1="${width - 135}"
                y1="26"
                x2="${width - 105}"
                y2="26"
                stroke="#77adf2"
                stroke-width="3"
                stroke-dasharray="6 5"
            />

            <text
                x="${width - 98}"
                y="30"
                fill="#aab4c2"
                font-size="11"
            >
                Predicted
            </text>


            <!-- Axis labels -->

            <text
                x="${width / 2}"
                y="${height - 15}"
                text-anchor="middle"
                fill="#748092"
                font-size="12"
            >
                Burn-In Time
            </text>

        </svg>
    `;
}


/* =========================================================
   UPDATE COMPLETE DASHBOARD
   ========================================================= */

function updateDashboard(response) {

    const data =
        normalizeAnalysis(response);

    console.log(
        "Analysis result:",
        data
    );

    updateOverview(data);

    updateAnomaly(data);

    updatePrediction(data);

    updateRisk(data);

    updateEngineeringLimit(data);

    updateMeasurements(data);

    updateReasons(data);

    drawBurnInChart(data);
}


/* =========================================================
   COMPONENT SELECTION
   ========================================================= */

async function handleComponentSelection() {

    const select =
        $("componentSelect");

    if (!select) {
        return;
    }

    const componentId =
        select.value;

    if (!componentId) {
        return;
    }

    setText(
        "componentStatus",
        `Analyzing ${componentId}...`
    );

    select.disabled = true;

    try {

        const response =
            await analyzeComponent(
                componentId
            );

        updateDashboard(
            response
        );

        setText(
            "componentStatus",
            `${componentId} analyzed successfully.`
        );

    } catch (error) {

        console.error(
            "Component analysis failed:",
            error
        );

        setText(
            "componentStatus",
            `Analysis failed: ${error.message}`
        );

        const body =
            $("overviewBody");

        if (body) {

            body.innerHTML = `
                <tr>
                    <td colspan="7">
                        Analysis failed:
                        ${escapeHtml(error.message)}
                    </td>
                </tr>
            `;
        }

    } finally {

        select.disabled = false;
    }
}


/* =========================================================
   REFRESH DASHBOARD
   ========================================================= */

async function refreshDashboard() {

    const button =
        $("refreshButton");

    if (button) {

        button.disabled = true;

        button.textContent =
            "Refreshing...";
    }

    try {

        await checkHealth();

        const components =
            await loadComponents();

        await loadSummaryCounts(
            components
        );

        const select =
            $("componentSelect");

        /*
           Preserve selected component if possible.
        */

        if (
            select &&
            select.value
        ) {

            await handleComponentSelection();
        }

    } finally {

        if (button) {

            button.disabled = false;

            button.textContent =
                "Refresh";
        }
    }
}


/* =========================================================
   INITIALIZE APPLICATION
   ========================================================= */

async function initializeApp() {

    console.log(
        "AegisBurn AI frontend starting..."
    );

    const refreshButton =
        $("refreshButton");

    const componentSelect =
        $("componentSelect");


    /*
       Register events.
    */

    if (refreshButton) {

        refreshButton.addEventListener(
            "click",
            refreshDashboard
        );
    }


    if (componentSelect) {

        componentSelect.addEventListener(
            "change",
            handleComponentSelection
        );
    }


    /*
       Check backend.
    */

    const healthy =
        await checkHealth();

    if (!healthy) {

        setText(
            "componentStatus",
            "Backend connection failed. Make sure Uvicorn is running on port 8001."
        );

        return;
    }


    /*
       Load component list.
    */

    const components =
        await loadComponents();


    /*
       Load dashboard totals.
    */

    await loadSummaryCounts(
        components
    );


    /*
       Automatically analyze the first component.
    */

    const select =
        $("componentSelect");

    if (
        select &&
        select.options.length > 1
    ) {

        const firstOption =
            Array.from(select.options)
                .find(
                    (option) =>
                        option.value
                );

        if (firstOption) {

            select.value =
                firstOption.value;

            await handleComponentSelection();
        }
    }

    console.log(
        "AegisBurn AI frontend ready."
    );
}


/* =========================================================
   START AFTER DOM LOAD
   ========================================================= */

if (
    document.readyState === "loading"
) {

    document.addEventListener(
        "DOMContentLoaded",
        initializeApp
    );

} else {

    initializeApp();
}