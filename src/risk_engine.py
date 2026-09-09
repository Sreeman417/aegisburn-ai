class RiskEngine:
    """
    AegisBurn AI risk engine.

    IMPORTANT:
    This engine performs inference ONLY from measured burn-in data,
    engineered anomaly features, and the model's prediction.

    Ground-truth dataset fields such as:
        - defect_type
        - is_defective

    are deliberately NOT used here.

    They may exist in a labeled evaluation dataset, but they must
    never influence a live screening decision.
    """

    def __init__(self):
        pass

    @staticmethod
    def _float(value, default=0.0):
        try:
            value = float(value)
            if value != value:
                return default
            return value
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value, low=0.0, high=100.0):
        return max(low, min(high, value))

    def evaluate_component(
        self,
        row,
        predicted_168h,
        safety_slope
    ):
        """
        Calculate component risk using ONLY inference-time information.

        Inputs used:
            anomaly_flag
            anomaly_score
            anomaly_index
            burn-in measurements
            engineered drift values
            predicted 168h value
            safety slope
            engineering limit

        Ground-truth labels are ignored.
        """

        # ==========================================================
        # BASIC INPUTS
        # ==========================================================

        anomaly_flag = bool(
            row.get("anomaly_flag", False)
        )

        anomaly_score = self._float(
            row.get(
                "anomaly_score",
                row.get("anomaly_index", 0.0)
            )
        )

        anomaly_index = self._float(
            row.get(
                "anomaly_index",
                anomaly_score
            )
        )

        predicted_168h = self._float(
            predicted_168h
        )

        safety_slope = self._float(
            safety_slope
        )

        value_0h = self._float(
            row.get("value_0h")
        )

        value_24h = self._float(
            row.get("value_24h")
        )

        value_96h = self._float(
            row.get("value_96h")
        )

        value_168h = self._float(
            row.get("value_168h")
        )

        engineering_limit = self._float(
            row.get("engineering_limit"),
            0.0
        )

        # ==========================================================
        # DERIVED DRIFT VALUES
        # ==========================================================

        drift_0_24 = self._float(
            row.get(
                "drift_0_24",
                value_24h - value_0h
            )
        )

        drift_24_96 = self._float(
            row.get(
                "drift_24_96",
                value_96h - value_24h
            )
        )

        drift_96_168 = self._float(
            row.get(
                "drift_96_168",
                value_168h - value_96h
            )
        )

        drift_0_168 = self._float(
            row.get(
                "drift_0_168",
                value_168h - value_0h
            )
        )

        early_slope = self._float(
            row.get(
                "slope_early",
                drift_0_24 / 24.0
            )
        )

        # ==========================================================
        # PREDICTED FUTURE DRIFT
        # ==========================================================

        future_drift = predicted_168h - value_24h

        if future_drift < 0:
            future_drift_for_risk = 0.0
        else:
            future_drift_for_risk = future_drift

        # ==========================================================
        # LIMIT UTILIZATION
        # ==========================================================

        limit_utilization = 0.0

        if engineering_limit > 0:
            observed_max = max(
                value_0h,
                value_24h,
                value_96h,
                value_168h,
                predicted_168h
            )

            limit_utilization = (
                observed_max /
                engineering_limit
            ) * 100.0

            limit_utilization = self._clamp(
                limit_utilization
            )

        # ==========================================================
        # RISK SCORE
        # ==========================================================

        risk_score = 0.0
        reasons = []

        # ----------------------------------------------------------
        # 1. AI ANOMALY DETECTION
        # ----------------------------------------------------------

        if anomaly_flag:
            risk_score += 45.0

            reasons.append(
                "AI anomaly detector identified abnormal burn-in behavior."
            )

            anomaly_extra = (
                self._clamp(anomaly_index) *
                0.30
            )

            risk_score += anomaly_extra

        elif anomaly_index >= 70:
            risk_score += 30.0

            reasons.append(
                "AI anomaly score indicates elevated abnormality."
            )

        elif anomaly_index >= 50:
            risk_score += 15.0

            reasons.append(
                "AI anomaly score indicates moderate abnormality."
            )

        # ----------------------------------------------------------
        # 2. EARLY DRIFT
        # ----------------------------------------------------------

        if value_0h != 0:

            early_relative_drift = (
                abs(drift_0_24) /
                abs(value_0h)
            ) * 100.0

        else:
            early_relative_drift = 0.0

        if early_relative_drift >= 20.0:

            risk_score += 20.0

            reasons.append(
                "Large parameter drift detected during the first 24 hours."
            )

        elif early_relative_drift >= 10.0:

            risk_score += 12.0

            reasons.append(
                "Elevated parameter drift detected during the first 24 hours."
            )

        elif early_relative_drift >= 5.0:

            risk_score += 6.0

            reasons.append(
                "Moderate early parameter drift detected."
            )

        # ----------------------------------------------------------
        # 3. LONG-TERM DRIFT
        # ----------------------------------------------------------

        if value_0h != 0:

            total_relative_drift = (
                abs(drift_0_168) /
                abs(value_0h)
            ) * 100.0

        else:
            total_relative_drift = 0.0

        if total_relative_drift >= 50.0:

            risk_score += 20.0

            reasons.append(
                "Severe parameter drift observed across the burn-in period."
            )

        elif total_relative_drift >= 30.0:

            risk_score += 14.0

            reasons.append(
                "Significant parameter drift observed across the burn-in period."
            )

        elif total_relative_drift >= 15.0:

            risk_score += 7.0

            reasons.append(
                "Moderate cumulative parameter drift detected."
            )

        # ----------------------------------------------------------
        # 4. FUTURE PREDICTION
        # ----------------------------------------------------------

        if value_24h != 0:

            predicted_relative_change = (
                future_drift_for_risk /
                abs(value_24h)
            ) * 100.0

        else:
            predicted_relative_change = 0.0

        if predicted_relative_change >= 50.0:

            risk_score += 15.0

            reasons.append(
                "Model predicts severe future parameter degradation."
            )

        elif predicted_relative_change >= 25.0:

            risk_score += 10.0

            reasons.append(
                "Model predicts significant future parameter degradation."
            )

        elif predicted_relative_change >= 10.0:

            risk_score += 5.0

            reasons.append(
                "Model predicts measurable future parameter drift."
            )

        # ----------------------------------------------------------
        # 5. ENGINEERING LIMIT
        # ----------------------------------------------------------

        if engineering_limit > 0:

            if limit_utilization >= 100.0:

                risk_score += 20.0

                reasons.append(
                    "Measured or predicted parameter exceeds the engineering limit."
                )

            elif limit_utilization >= 85.0:

                risk_score += 15.0

                reasons.append(
                    "Parameter is approaching the engineering limit."
                )

            elif limit_utilization >= 70.0:

                risk_score += 8.0

                reasons.append(
                    "Parameter utilization of the engineering limit is elevated."
                )

        # ==========================================================
        # 6. SAFETY SLOPE
        # ==========================================================

        # Positive slope means the parameter is increasing.
        # For the prototype parameter families, increasing values
        # represent degradation.

        if safety_slope > 0:

            if value_0h != 0:

                slope_relative = (
                    abs(safety_slope) /
                    abs(value_0h)
                ) * 100.0

            else:
                slope_relative = 0.0

            if slope_relative >= 1.0:

                risk_score += 10.0

                reasons.append(
                    "Positive degradation slope indicates continuing parameter growth."
                )

            elif slope_relative >= 0.25:

                risk_score += 5.0

                reasons.append(
                    "Burn-in measurements show a continuing degradation trend."
                )

        # ==========================================================
        # FINAL SCORE
        # ==========================================================

        risk_score = self._clamp(
            risk_score
        )

        # ==========================================================
        # RISK LEVEL
        # ==========================================================

        if risk_score >= 80:

            risk_level = "CRITICAL"

        elif risk_score >= 60:

            risk_level = "HIGH"

        elif risk_score >= 30:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        # ==========================================================
        # DECISION
        # ==========================================================

        if risk_level == "CRITICAL":

            recommendation = (
                "FLAG COMPONENT — severe degradation or anomaly indicators "
                "detected. Immediate inspection recommended."
            )

        elif risk_level == "HIGH":

            recommendation = (
                "FLAG COMPONENT — significant abnormal behavior detected. "
                "Additional inspection recommended."
            )

        elif risk_level == "MEDIUM":

            recommendation = (
                "REVIEW COMPONENT — abnormal drift indicators detected. "
                "Continue monitoring and review."
            )

        else:

            recommendation = (
                "PASS / MONITOR — no significant abnormal behavior detected "
                "from the available burn-in measurements."
            )

        # ==========================================================
        # FALLBACK EXPLANATION
        # ==========================================================

        if not reasons:

            reasons.append(
                "No significant anomaly, drift, prediction, or limit indicators detected."
            )

        # Remove duplicate reasons while preserving order.

        reasons = list(
            dict.fromkeys(reasons)
        )

        # ==========================================================
        # RETURN
        # ==========================================================

        return {
            "risk_score": round(
                risk_score,
                2
            ),

            "risk_level": risk_level,

            "anomaly_flag": anomaly_flag,

            "anomaly_score": round(
                anomaly_score,
                4
            ),

            "anomaly_index": round(
                self._clamp(anomaly_index),
                2
            ),

            "predicted_168h": round(
                predicted_168h,
                4
            ),

            "safety_slope": round(
                safety_slope,
                6
            ),

            "early_drift": round(
                drift_0_24,
                4
            ),

            "future_drift": round(
                future_drift,
                4
            ),

            "early_drift_index": round(
                self._clamp(
                    early_relative_drift
                ),
                2
            ),

            "future_drift_index": round(
                self._clamp(
                    predicted_relative_change
                ),
                2
            ),

            "limit_utilization": round(
                limit_utilization,
                2
            ),

            "engineering_limit": round(
                engineering_limit,
                4
            ),

            "risk_reasons": reasons,

            "reasons": reasons,

            "risk_decision": risk_level,

            "recommendation": recommendation,
        }