class RiskEngine:
    """
    AegisBurn AI risk engine.

    Combines anomaly detection results, future-drift predictions, and
    safety-slope information into a single risk score/level for a
    component.

    This version is safe even when anomaly models are not loaded.
    Missing anomaly fields automatically default to safe values.
    """

    def __init__(self):
        pass

    def evaluate_component(
        self,
        row,
        predicted_168h,
        safety_slope
    ):
        """
        Evaluate the risk of a component.

        This version is safe even when anomaly models are not loaded.
        Missing anomaly fields automatically default to safe values.
        """

        # ==========================================================
        # SAFE INPUT EXTRACTION
        # ==========================================================

        anomaly_flag = bool(
            row.get("anomaly_flag", False)
        )

        anomaly_score = row.get(
            "anomaly_score",
            row.get("anomaly_severity", 0.0)
        )

        try:
            anomaly_score = float(anomaly_score)
        except (TypeError, ValueError):
            anomaly_score = 0.0


        # ==========================================================
        # DATASET DEFECT FALLBACK
        # ==========================================================

        is_defective_value = str(
            row.get("is_defective", 0)
        ).strip().lower()

        is_defective = is_defective_value in [
            "1",
            "true",
            "yes",
            "defective"
        ]

        # If the dataset says defective, force anomaly detection
        if is_defective:
            anomaly_flag = True

            if anomaly_score <= 0:
                anomaly_score = 1.0


        # ==========================================================
        # SAFE NUMERIC VALUES
        # ==========================================================

        try:
            predicted_168h = float(predicted_168h)
        except (TypeError, ValueError):
            predicted_168h = 0.0

        try:
            safety_slope = float(safety_slope)
        except (TypeError, ValueError):
            safety_slope = 0.0


        # ==========================================================
        # RISK SCORING
        # ==========================================================

        risk_score = 0.0


        # ----------------------------------------------------------
        # ANOMALY RISK
        # ----------------------------------------------------------

        if anomaly_flag:

            # Base anomaly contribution
            risk_score += 40.0

            # Additional contribution from anomaly score
            anomaly_contribution = anomaly_score * 30.0

            if anomaly_contribution > 30.0:
                anomaly_contribution = 30.0

            risk_score += anomaly_contribution


        # ----------------------------------------------------------
        # SAFETY SLOPE RISK
        # ----------------------------------------------------------

        # Negative slope can indicate degradation
        if safety_slope < 0:

            slope_risk = abs(safety_slope) * 10.0

            if slope_risk > 20.0:
                slope_risk = 20.0

            risk_score += slope_risk


        # ----------------------------------------------------------
        # PREDICTION RISK
        # ----------------------------------------------------------

        try:

            current_value = float(
                row.get(
                    "value_0h",
                    0.0
                )
            )

        except (TypeError, ValueError):

            current_value = 0.0


        # Compare predicted value with current value
        if current_value != 0:

            prediction_change = abs(
                predicted_168h - current_value
            )

            relative_change = (
                prediction_change /
                abs(current_value)
            )

            prediction_risk = relative_change * 20.0

            if prediction_risk > 20.0:
                prediction_risk = 20.0

            risk_score += prediction_risk


        # ==========================================================
        # LIMIT SCORE
        # ==========================================================

        risk_score = max(
            0.0,
            min(
                100.0,
                risk_score
            )
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
        # RECOMMENDATION
        # ==========================================================

        if risk_level == "CRITICAL":

            recommendation = (
                "Immediate inspection recommended. "
                "Component shows severe risk indicators."
            )

        elif risk_level == "HIGH":

            recommendation = (
                "Schedule inspection as soon as possible. "
                "Component shows significant risk indicators."
            )

        elif risk_level == "MEDIUM":

            recommendation = (
                "Monitor the component closely. "
                "Risk indicators require attention."
            )

        else:

            recommendation = (
                "Component appears stable. "
                "Continue normal monitoring."
            )


        # ==========================================================
        # RETURN RESULT
        # ==========================================================

        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,

            "anomaly_flag": anomaly_flag,
            "anomaly_score": round(anomaly_score, 4),

            "predicted_168h": round(
                predicted_168h,
                4
            ),

            "safety_slope": round(
                safety_slope,
                6
            ),

            "recommendation": recommendation
        }
