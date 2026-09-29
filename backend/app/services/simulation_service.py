from __future__ import annotations

import datetime as dt
import logging
import math
import uuid
from typing import Any

import numpy as np
import pandas as pd

from backend.app.llm.gemini_provider import GeminiProvider
from backend.app.llm.prompts import build_simulation_evidence
from backend.app.llm.provider import LLMError, LLMProvider
from backend.app.schemas.simulations import (
    AnomalyContext,
    DailySimulationResult,
    ForecastDelta,
    ForecastSummary,
    ScenarioType,
    SimulationRequest,
    SimulationResponse,
    SimulationStatus,
)
from backend.app.services.forecast_service import (
    DATASET_PATH,
    MODEL_CONFIG,
    BackendForecastService,
)
from backend.app.services.investigation_service import InvestigationService

logger = logging.getLogger(__name__)


class UnsupportedScenarioError(ValueError):
    """Raised when a requested scenario is not supported by current forecasting models."""

    pass



def validate_simulation_result(response: SimulationResponse) -> None:
    """
    Enforce 10 numerical and structural invariants on completed simulation responses.
    Raises ValueError if any consistency rule is violated.
    """
    if response.status != SimulationStatus.COMPLETED:
        return

    if response.baseline is None or response.scenario is None or response.delta is None:
        raise ValueError("Completed simulation must include baseline, scenario, and delta summaries.")

    if len(response.daily_results) != response.horizon_days:
        raise ValueError(
            f"daily_results length ({len(response.daily_results)}) does not match horizon_days ({response.horizon_days})."
        )

    # 1. Sum of baseline daily quantities matches summary total
    sum_b_qty = sum(d.baseline_quantity for d in response.daily_results)
    if abs(sum_b_qty - response.baseline.total_quantity) > 0.05:
        raise ValueError(
            f"Baseline quantity sum ({sum_b_qty:.2f}) does not match summary ({response.baseline.total_quantity:.2f})."
        )

    # 2. Sum of scenario daily quantities matches summary total
    sum_s_qty = sum(d.scenario_quantity for d in response.daily_results)
    if abs(sum_s_qty - response.scenario.total_quantity) > 0.05:
        raise ValueError(
            f"Scenario quantity sum ({sum_s_qty:.2f}) does not match summary ({response.scenario.total_quantity:.2f})."
        )

    # 3 & 4. Revenue sums match summary totals if revenue included
    if response.baseline.total_revenue is not None:
        sum_b_rev = sum(d.baseline_revenue or 0.0 for d in response.daily_results)
        if abs(sum_b_rev - response.baseline.total_revenue) > 0.05:
            raise ValueError(
                f"Baseline revenue sum ({sum_b_rev:.2f}) does not match summary ({response.baseline.total_revenue:.2f})."
            )

        sum_s_rev = sum(d.scenario_revenue or 0.0 for d in response.daily_results)
        if abs(sum_s_rev - response.scenario.total_revenue) > 0.05:
            raise ValueError(
                f"Scenario revenue sum ({sum_s_rev:.2f}) does not match summary ({response.scenario.total_revenue:.2f})."
            )

    # 5. Delta quantity matches scenario - baseline
    expected_qty_delta = response.scenario.total_quantity - response.baseline.total_quantity
    if abs(response.delta.quantity_delta - expected_qty_delta) > 0.05:
        raise ValueError(
            f"Quantity delta ({response.delta.quantity_delta:.2f}) does not match scenario - baseline ({expected_qty_delta:.2f})."
        )

    # 6. Delta percentage matches relative shift
    if response.baseline.total_quantity > 0:
        expected_qty_pct = (response.delta.quantity_delta / response.baseline.total_quantity) * 100.0
        if abs(response.delta.quantity_delta_percent - expected_qty_pct) > 0.1:
            raise ValueError(
                f"Quantity delta % ({response.delta.quantity_delta_percent:.2f}) does not match relative shift ({expected_qty_pct:.2f}%)."
            )

    # 7. Delta revenue matches scenario - baseline
    if response.scenario.total_revenue is not None and response.baseline.total_revenue is not None:
        expected_rev_delta = response.scenario.total_revenue - response.baseline.total_revenue
        if response.delta.revenue_delta is not None and abs(response.delta.revenue_delta - expected_rev_delta) > 0.05:
            raise ValueError(
                f"Revenue delta ({response.delta.revenue_delta:.2f}) does not match scenario - baseline ({expected_rev_delta:.2f})."
            )

    # 8. No negative quantity in daily results
    for d in response.daily_results:
        if d.baseline_quantity < 0:
            raise ValueError(f"Negative baseline quantity ({d.baseline_quantity}) on date {d.date}.")
        if d.scenario_quantity < 0:
            raise ValueError(f"Negative scenario quantity ({d.scenario_quantity}) on date {d.date}.")

    # 9. No NaN or infinite values
    for d in response.daily_results:
        for val in (d.baseline_quantity, d.scenario_quantity, d.delta_quantity, d.delta_percent):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Invalid non-finite daily value {val} on date {d.date}.")

    # 10. Sequential continuous dates
    for i in range(1, len(response.daily_results)):
        prev_date = response.daily_results[i - 1].date
        curr_date = response.daily_results[i].date
        if curr_date != prev_date + dt.timedelta(days=1):
            raise ValueError(f"Discontinuous date sequence between {prev_date} and {curr_date}.")


_SIMULATION_CACHE: dict[str, SimulationResponse] = {}


class SimulationService:
    """
    Production service for what-if scenario simulation.
    Projects hypothetical demand shifts, temporary shocks, trends, and event scenarios
    over 7, 30, and 90 day horizons without altering historical sales data.
    """

    def __init__(
        self,
        forecast_service: BackendForecastService | None = None,
        investigation_service: InvestigationService | None = None,
        llm_provider: LLMProvider | None = None,
        cache: dict[str, SimulationResponse] | None = None,
        elasticity_model: Any = None,
    ) -> None:
        self.forecast_service = forecast_service or BackendForecastService()
        self.investigation_service = investigation_service or InvestigationService()
        self.llm_provider = llm_provider or GeminiProvider()
        self._cache: dict[str, SimulationResponse] = cache if cache is not None else _SIMULATION_CACHE
        self._mock_elasticity_model = elasticity_model

    def get_simulation(self, simulation_id: str) -> SimulationResponse | None:
        """Retrieve a cached simulation result by ID."""
        return self._cache.get(simulation_id)

    def _get_active_elasticity_model(
        self,
        user_id: int | None = None,
        db: Any = None,
    ) -> Any:
        """Loads active ready elasticity model for user_id, with mock support."""
        if self._mock_elasticity_model is not None:
            return self._mock_elasticity_model
        if db is None or user_id is None:
            return None
        try:
            from sqlalchemy import select
            from backend.app.database.models import CompanyElasticityModel

            stmt = (
                select(CompanyElasticityModel)
                .where(
                    CompanyElasticityModel.user_id == user_id,
                    CompanyElasticityModel.is_active == True,
                    CompanyElasticityModel.status == "ready",
                )
                .order_by(CompanyElasticityModel.created_at.desc())
            )
            return db.scalars(stmt).first()
        except Exception as exc:
            logger.warning("Could not query active elasticity model: %s", exc)
            return None


    def _get_baseline_realized_price(
        self,
        anomaly_id: str | None = None,
        user_id: int | None = None,
        db: Any = None,
    ) -> float:
        """
        Derives baseline realized unit price (₹/unit).
        If anchored to an anomaly with price driver metrics, uses that reference;
        otherwise computes 28-day historical aggregate price realization.
        """
        if anomaly_id:
            try:
                investigation = self.investigation_service.investigate_anomaly(
                    anomaly_id, user_id=user_id, db=db
                )
                for driver in investigation.drivers:
                    if driver.driver_type == "price" and driver.reference_value and driver.reference_value > 0:
                        return round(float(driver.reference_value), 2)
            except Exception as exc:
                logger.warning("Could not derive price driver from anomaly %s: %s", anomaly_id, exc)

        # Fallback to historical 28-day dataset realized price
        try:
            from backend.app.services.dataset_runtime_service import (
                dataset_runtime_service,
            )
            df = dataset_runtime_service.get_daily_aggregate(user_id=user_id, db=db)
            if not df.empty and "Sales_Amount" in df.columns and "Quantity" in df.columns:
                recent_sales = df["Sales_Amount"].tail(28).sum()
                recent_qty = df["Quantity"].tail(28).sum()
                if recent_qty > 0:
                    return round(float(recent_sales / recent_qty), 2)
        except Exception as exc:
            logger.warning("Could not read recent realized price from dataset: %s", exc)

        return 3400.87

    def _generate_deterministic_explanation(
        self,
        response: SimulationResponse,
        unit_price: float,
        scenario_unit_price: float | None = None,
    ) -> str:
        """Generates deterministic, executive-ready explanation of pre-calculated simulation results."""
        if response.baseline is None or response.scenario is None or response.delta is None:
            return "Simulation could not be executed."

        horizon = response.horizon_days
        stype = response.scenario_type.value.replace("_", " ").title()
        qty_delta = response.delta.quantity_delta
        qty_pct = response.delta.quantity_delta_percent
        scenario_qty = response.scenario.total_quantity
        baseline_qty = response.baseline.total_quantity

        explanation_parts = [
            f"Under a hypothetical {stype} across a {horizon}-day horizon, projected total demand is {scenario_qty:,.1f} units "
            f"compared to a baseline of {baseline_qty:,.1f} units (variance of {qty_delta:+,.1f} units, {qty_pct:+.1f}%)."
        ]

        if response.scenario.total_revenue is not None and response.baseline.total_revenue is not None and response.delta.revenue_delta is not None:
            scenario_rev = response.scenario.total_revenue
            baseline_rev = response.baseline.total_revenue
            rev_delta = response.delta.revenue_delta
            rev_pct = response.delta.revenue_delta_percent or 0.0
            if response.scenario_type == ScenarioType.PRICE_CHANGE and scenario_unit_price is not None:
                explanation_parts.append(
                    f"Projected revenue is ₹{scenario_rev:,.2f} versus baseline ₹{baseline_rev:,.2f} "
                    f"({rev_delta:+,.2f}, {rev_pct:+.1f}%), evaluating price adjustment from ₹{unit_price:,.2f} to ₹{scenario_unit_price:,.2f}."
                )
            elif response.scenario_type == ScenarioType.DISCOUNT_CHANGE and scenario_unit_price is not None:
                explanation_parts.append(
                    f"Projected revenue is ₹{scenario_rev:,.2f} versus baseline ₹{baseline_rev:,.2f} "
                    f"({rev_delta:+,.2f}, {rev_pct:+.1f}%), evaluating effective markdown selling price of ₹{scenario_unit_price:,.2f}."
                )
            else:
                explanation_parts.append(
                    f"Projected revenue is ₹{scenario_rev:,.2f} versus baseline ₹{baseline_rev:,.2f} "
                    f"({rev_delta:+,.2f}, {rev_pct:+.1f}%), assuming a constant realized unit price of ₹{unit_price:,.2f}."
                )

        if response.scenario_type == ScenarioType.PRICE_CHANGE and response.price_elasticity is not None:
            explanation_parts.append(
                f"Demand sensitivity was evaluated using estimated historical price elasticity of {response.price_elasticity:+.3f} (observational correlation; not guaranteed causal effect)."
            )
        elif response.scenario_type == ScenarioType.DISCOUNT_CHANGE and response.discount_sensitivity is not None:
            explanation_parts.append(
                f"Demand sensitivity was evaluated using estimated historical discount sensitivity coefficient of {response.discount_sensitivity:+.3f} (observational correlation; not guaranteed causal effect)."
            )

        explanation_parts.append(
            "This simulation represents an exploratory planning scenario and requires qualified operational review before decision-making."
        )

        return " ".join(explanation_parts)

    def run_simulation(
        self,
        request: SimulationRequest,
        user_id: int | None = None,
        db: Any = None,
    ) -> SimulationResponse:
        """
        Execute what-if simulation based on request specifications.
        """
        sim_id = f"sim-{uuid.uuid4().hex[:12]}"
        model_name = MODEL_CONFIG[request.horizon_days]["model_name"]

        # 1. Handle Elasticity Scenarios: price_change and discount_change
        elasticity_model: Any = None
        if request.scenario_type in (ScenarioType.PRICE_CHANGE, ScenarioType.DISCOUNT_CHANGE):
            elasticity_model = self._get_active_elasticity_model(user_id=user_id, db=db)
            if not elasticity_model:
                # Check if there is an inactive/insufficient model in DB to provide the precise diagnostic reason
                if db is not None and user_id is not None:
                    try:
                        from sqlalchemy import select
                        from backend.app.database.models import CompanyElasticityModel

                        latest_model = db.scalars(
                            select(CompanyElasticityModel)
                            .where(CompanyElasticityModel.user_id == user_id)
                            .order_by(CompanyElasticityModel.created_at.desc())
                        ).first()
                        if latest_model and latest_model.status in ("unavailable", "insufficient_data"):
                            diag = latest_model.diagnostics or {}
                            if request.scenario_type == ScenarioType.PRICE_CHANGE and diag.get("price_reason"):
                                raise UnsupportedScenarioError(diag["price_reason"])
                            if request.scenario_type == ScenarioType.DISCOUNT_CHANGE and diag.get("discount_reason"):
                                raise UnsupportedScenarioError(diag["discount_reason"])
                            if latest_model.status_message:
                                raise UnsupportedScenarioError(latest_model.status_message)
                    except UnsupportedScenarioError:
                        raise
                    except Exception:
                        pass
                raise UnsupportedScenarioError(
                    "This scenario requires a dedicated elasticity model and is not supported by the current forecasting model."
                )

            # Check specific feature capability on the ready model
            diag = getattr(elasticity_model, "diagnostics", None) or {}
            if request.scenario_type == ScenarioType.PRICE_CHANGE:
                price_supported = diag.get("price_supported", getattr(elasticity_model, "price_elasticity", None) is not None)
                if not price_supported:
                    raise UnsupportedScenarioError(
                        diag.get("price_reason")
                        or "Price Change is unavailable because the active dataset does not contain sufficient historical price variation to estimate demand sensitivity."
                    )
            elif request.scenario_type == ScenarioType.DISCOUNT_CHANGE:
                discount_supported = diag.get("discount_supported", getattr(elasticity_model, "discount_sensitivity", None) is not None)
                if not discount_supported:
                    raise UnsupportedScenarioError(
                        diag.get("discount_reason")
                        or "Discount Depth is unavailable because the active dataset does not contain sufficient historical discount variation to estimate discount sensitivity."
                    )



        # 2. Extract Anomaly Context if anchored
        anomaly_context: AnomalyContext | None = None
        if request.anomaly_id:
            try:
                inv = self.investigation_service.investigate_anomaly(
                    request.anomaly_id, user_id=user_id, db=db
                )
                key_contribs = [
                    f"{d.driver_name}: {d.difference_percent:+.1f}% shift ({d.driver_type})"
                    if d.difference_percent is not None
                    else f"{d.driver_name} ({d.driver_type})"
                    for d in inv.drivers[:3]
                ]
                anomaly_context = AnomalyContext(
                    anomaly_id=inv.anomaly_id,
                    anomaly_date=inv.anomaly_date,
                    metric=inv.metric,
                    actual_value=inv.actual_value,
                    expected_baseline=inv.expected_baseline,
                    deviation=inv.deviation,
                    deviation_percent=inv.deviation_percent,
                    severity=inv.severity,
                    key_contributors=key_contribs,
                )
            except Exception as exc:
                logger.warning("Failed to retrieve anomaly context for %s: %s", request.anomaly_id, exc)

        # 3. Generate Baseline Forecast
        try:
            _, baseline_df = self.forecast_service.generate_forecast(
                horizon=request.horizon_days, user_id=user_id, db=db
            )
        except Exception as exc:
            logger.error("Failed to generate baseline forecast: %s", exc)
            return SimulationResponse(
                status=SimulationStatus.INVALID_REQUEST,
                simulation_id=sim_id,
                anomaly_id=request.anomaly_id,
                scenario_type=request.scenario_type,
                horizon_days=request.horizon_days,
                model_name=model_name,
                confidence="not_available",
                reason=f"Baseline forecast generation failed: {exc}",
            )

        baseline_quantities = baseline_df["Predicted_Quantity"].astype(float).values
        forecast_dates = pd.to_datetime(baseline_df["Date"]).dt.date.values
        unit_price = self._get_baseline_realized_price(
            request.anomaly_id, user_id=user_id, db=db
        )

        assumptions: list[str] = []
        limitations: list[str] = [
            "Simulation is a hypothetical exploratory projection and does not constitute a guaranteed outcome.",
            "Operational constraints (warehouse capacity, backorders, supplier lead times) require human verification.",
        ]
        confidence: Any = "high"

        # 4. Generate Scenario Forecast per Scenario Type
        scenario_quantities = np.zeros_like(baseline_quantities)
        scenario_unit_price = unit_price

        if request.scenario_type == ScenarioType.DEMAND_MULTIPLIER:
            pct = request.demand_change_percent if request.demand_change_percent is not None else 0.0
            mult = 1.0 + (pct / 100.0)
            scenario_quantities = np.maximum(0.0, baseline_quantities * mult)
            confidence = "high"
            assumptions.append(
                f"Demand across the entire {request.horizon_days}-day horizon is uniformly adjusted by {pct:+.1f}% relative to baseline."
            )
            limitations.append("Assumes uniform percentage elasticity across all product lines, channels, and customer segments.")

        elif request.scenario_type == ScenarioType.TEMPORARY_SHOCK:
            pct = request.demand_change_percent if request.demand_change_percent is not None else 0.0
            duration = min(request.shock_duration_days or 7, request.horizon_days)
            mult = 1.0 + (pct / 100.0)
            scenario_quantities = baseline_quantities.copy()
            scenario_quantities[:duration] = np.maximum(0.0, baseline_quantities[:duration] * mult)
            confidence = "high"
            assumptions.append(
                f"Temporary demand shift of {pct:+.1f}% applies for the initial {duration} days, returning to baseline forecast thereafter."
            )
            limitations.append("Assumes instantaneous return to baseline with zero residual churn, backlog, or customer hysteresis.")

        elif request.scenario_type == ScenarioType.PERSISTENT_SHIFT:
            pct = request.demand_change_percent if request.demand_change_percent is not None else 0.0
            mult = 1.0 + (pct / 100.0)
            scenario_quantities = np.maximum(0.0, baseline_quantities * mult)
            confidence = "medium"
            assumptions.append(
                f"Persistent structural demand shift of {pct:+.1f}% sustained across all {request.horizon_days} days."
            )
            limitations.append("Assumes persistent market realignment without capacity constraints or competitor countermeasures.")

        elif request.scenario_type == ScenarioType.TREND_CONTINUATION:
            # Check historical data length
            if not DATASET_PATH.exists():
                return SimulationResponse(
                    status=SimulationStatus.INSUFFICIENT_DATA,
                    simulation_id=sim_id,
                    anomaly_id=request.anomaly_id,
                    scenario_type=request.scenario_type,
                    horizon_days=request.horizon_days,
                    model_name=model_name,
                    confidence="not_available",
                    reason="Historical sales dataset not found for trend continuation analysis.",
                )

            hist_df = pd.read_csv(DATASET_PATH, usecols=["Quantity"])
            if len(hist_df) < request.trend_window_days:
                return SimulationResponse(
                    status=SimulationStatus.INSUFFICIENT_DATA,
                    simulation_id=sim_id,
                    anomaly_id=request.anomaly_id,
                    scenario_type=request.scenario_type,
                    horizon_days=request.horizon_days,
                    model_name=model_name,
                    confidence="not_available",
                    reason=(
                        f"Historical dataset has {len(hist_df)} observations, which is fewer than "
                        f"the required trend window of {request.trend_window_days} days."
                    ),
                )

            recent_hist = hist_df["Quantity"].tail(request.trend_window_days).astype(float).values
            x = np.arange(request.trend_window_days)
            slope, _ = np.polyfit(x, recent_hist, 1)

            # Project linear trend velocity forward relative to baseline
            steps = np.arange(1, request.horizon_days + 1)
            scenario_quantities = np.maximum(0.0, baseline_quantities + (slope * steps))
            confidence = "medium"
            assumptions.append(
                f"Recent {request.trend_window_days}-day historical linear trend (slope: {slope:+.2f} units/day) continues cumulatively over the {request.horizon_days}-day horizon."
            )
            limitations.append("Linear trend extrapolation assumes unconstrained trajectory without market saturation or seasonal reversal.")

        elif request.scenario_type in (ScenarioType.PROMOTION_SCENARIO, ScenarioType.HOLIDAY_SCENARIO):
            # Use production ML forecast service with simulated events
            date_index = pd.DatetimeIndex(forecast_dates)
            promo_val = 1.0 if (request.scenario_type == ScenarioType.PROMOTION_SCENARIO and request.promotion_active) else 0.0
            holiday_val = 1.0 if (request.scenario_type == ScenarioType.HOLIDAY_SCENARIO and request.holiday_active) else 0.0

            events = pd.DataFrame(
                {
                    "Promotions": [promo_val] * request.horizon_days,
                    "Holiday_Flag": [holiday_val] * request.horizon_days,
                },
                index=date_index,
            )

            try:
                _, event_forecast_df = self.forecast_service.generate_forecast(
                    horizon=request.horizon_days, events=events
                )
                scenario_quantities = np.maximum(
                    0.0, event_forecast_df["Predicted_Quantity"].astype(float).values
                )
            except Exception as exc:
                logger.error("Failed to run event-based ML forecast: %s", exc)
                return SimulationResponse(
                    status=SimulationStatus.INVALID_REQUEST,
                    simulation_id=sim_id,
                    anomaly_id=request.anomaly_id,
                    scenario_type=request.scenario_type,
                    horizon_days=request.horizon_days,
                    model_name=model_name,
                    confidence="not_available",
                    reason=f"Event forecast execution failed: {exc}",
                )

            if request.scenario_type == ScenarioType.PROMOTION_SCENARIO:
                status_str = "active (1.0)" if request.promotion_active else "inactive (0.0)"
                assumptions.append(
                    f"Promotional campaign flag set to {status_str} across all {request.horizon_days} forecast days."
                )
                limitations.append(
                    f"Event-driven ML forecast reflects historical promotional lift learned by {model_name}; campaign-specific creative and spend variations are not represented."
                )
            else:
                status_str = "active (1.0)" if request.holiday_active else "inactive (0.0)"
                assumptions.append(
                    f"Holiday trading flag set to {status_str} across all {request.horizon_days} forecast days."
                )
                limitations.append(
                    f"Calendar holiday impact reflects aggregate historical calendar lift without specific festive promotional bundling."
                )
            confidence = "high"

        elif request.scenario_type == ScenarioType.PRICE_CHANGE:
            pct = request.price_change_percent if request.price_change_percent is not None else (
                request.demand_change_percent if request.demand_change_percent is not None else 0.0
            )
            p0 = unit_price
            p1 = round(p0 * (1.0 + pct / 100.0), 2)
            if p1 <= 0:
                raise ValueError(f"Scenario unit price must be strictly positive (computed ₹{p1:.2f} from {pct:+.1f}% shift).")

            beta_p = getattr(elasticity_model, "price_elasticity", -1.0)
            if beta_p is None:
                beta_p = -1.0

            price_ratio = 1.0 + (pct / 100.0)
            mult = math.pow(price_ratio, beta_p)
            scenario_quantities = np.maximum(0.0, baseline_quantities * mult)
            scenario_unit_price = p1
            model_name = f"Elasticity (RidgeLogLog v{getattr(elasticity_model, 'model_version', 1)})"
            confidence = "high" if abs(pct) <= 20.0 else "medium"

            assumptions.append(
                f"Unit price shifted by {pct:+.1f}% relative to baseline (from ₹{p0:,.2f} to ₹{p1:,.2f})."
            )
            assumptions.append(
                f"Demand adjusted via company elasticity model v{getattr(elasticity_model, 'model_version', 1)} "
                f"(estimated price elasticity: {beta_p:+.4f})."
            )
            limitations.append(
                "Estimated historical price sensitivity reflects observational correlation and does not guarantee causal market outcomes."
            )

        elif request.scenario_type == ScenarioType.DISCOUNT_CHANGE:
            pct = request.discount_change_percent if request.discount_change_percent is not None else (
                request.demand_change_percent if request.demand_change_percent is not None else 0.0
            )
            base_discount = 0.0
            scenario_discount = min(100.0, max(0.0, base_discount + pct))

            beta_d = getattr(elasticity_model, "discount_sensitivity", 0.5)
            if beta_d is None:
                beta_d = 0.5

            delta_norm = (scenario_discount - base_discount) / 100.0
            mult = math.exp(beta_d * delta_norm)
            scenario_quantities = np.maximum(0.0, baseline_quantities * mult)

            # Effective selling prices
            eff_base = round(unit_price * (1.0 - base_discount / 100.0), 2)
            eff_scen = round(unit_price * (1.0 - scenario_discount / 100.0), 2)
            if eff_scen < 0:
                eff_scen = 0.0

            unit_price = eff_base
            scenario_unit_price = eff_scen
            model_name = f"Elasticity (RidgeLogLog v{getattr(elasticity_model, 'model_version', 1)})"
            confidence = "high" if abs(pct) <= 20.0 else "medium"

            assumptions.append(
                f"Discount depth changed by {pct:+.1f} percentage points ({base_discount:.1f}% -> {scenario_discount:.1f}%). "
                f"Effective selling price shifted from ₹{eff_base:,.2f} to ₹{eff_scen:,.2f}."
            )
            assumptions.append(
                f"Demand adjusted via company elasticity model v{getattr(elasticity_model, 'model_version', 1)} "
                f"(estimated discount sensitivity: {beta_d:+.4f})."
            )
            limitations.append(
                "Estimated discount sensitivity reflects observational historical promotion markdown response without basket cross-elasticity."
            )

        if request.include_revenue:
            if request.scenario_type == ScenarioType.PRICE_CHANGE:
                assumptions.append(
                    f"Revenue calculated under dynamic scenario unit price (baseline: ₹{unit_price:,.2f}, scenario: ₹{scenario_unit_price:,.2f})."
                )
            elif request.scenario_type == ScenarioType.DISCOUNT_CHANGE:
                assumptions.append(
                    f"Revenue calculated under dynamic effective markdown price (baseline: ₹{unit_price:,.2f}, scenario: ₹{scenario_unit_price:,.2f})."
                )
            else:
                assumptions.append(
                    f"Revenue calculated under constant baseline realized unit price assumption (₹{unit_price:,.2f} per unit). Price elasticity and customer margin dynamics are not modeled."
                )
                limitations.append("Price elasticity and profit margin trade-offs must be validated separately.")

        # 5. Build Daily Simulation Results
        daily_results: list[DailySimulationResult] = []
        for i in range(request.horizon_days):
            d_date = forecast_dates[i]
            b_q = round(float(baseline_quantities[i]), 2)
            s_q = round(float(scenario_quantities[i]), 2)
            d_q = round(s_q - b_q, 2)
            d_pct = round((d_q / b_q * 100.0), 2) if b_q > 0 else 0.0

            b_r: float | None = None
            s_r: float | None = None
            d_r: float | None = None
            if request.include_revenue:
                b_r = round(b_q * unit_price, 2)
                s_r = round(s_q * scenario_unit_price, 2)
                d_r = round(s_r - b_r, 2)

            daily_results.append(
                DailySimulationResult(
                    date=d_date,
                    baseline_quantity=b_q,
                    scenario_quantity=s_q,
                    delta_quantity=d_q,
                    delta_percent=d_pct,
                    baseline_revenue=b_r,
                    scenario_revenue=s_r,
                    delta_revenue=d_r,
                )
            )

        # 6. Build Summaries
        total_b_qty = round(sum(d.baseline_quantity for d in daily_results), 2)
        total_s_qty = round(sum(d.scenario_quantity for d in daily_results), 2)
        avg_b_qty = round(total_b_qty / request.horizon_days, 2)
        avg_s_qty = round(total_s_qty / request.horizon_days, 2)

        total_b_rev: float | None = None
        total_s_rev: float | None = None
        avg_b_rev: float | None = None
        avg_s_rev: float | None = None
        qty_delta = round(total_s_qty - total_b_qty, 2)
        qty_delta_pct = round((qty_delta / total_b_qty * 100.0), 2) if total_b_qty > 0 else 0.0
        rev_delta: float | None = None
        rev_delta_pct: float | None = None

        if request.include_revenue:
            total_b_rev = round(sum(d.baseline_revenue or 0.0 for d in daily_results), 2)
            total_s_rev = round(sum(d.scenario_revenue or 0.0 for d in daily_results), 2)
            avg_b_rev = round(total_b_rev / request.horizon_days, 2)
            avg_s_rev = round(total_s_rev / request.horizon_days, 2)
            rev_delta = round(total_s_rev - total_b_rev, 2)
            rev_delta_pct = round((rev_delta / total_b_rev * 100.0), 2) if total_b_rev > 0 else 0.0

        baseline_summary = ForecastSummary(
            total_quantity=total_b_qty,
            total_revenue=total_b_rev,
            average_daily_quantity=avg_b_qty,
            average_daily_revenue=avg_b_rev,
        )

        scenario_summary = ForecastSummary(
            total_quantity=total_s_qty,
            total_revenue=total_s_rev,
            average_daily_quantity=avg_s_qty,
            average_daily_revenue=avg_s_rev,
        )

        delta_summary = ForecastDelta(
            quantity_delta=qty_delta,
            quantity_delta_percent=qty_delta_pct,
            revenue_delta=rev_delta,
            revenue_delta_percent=rev_delta_pct,
        )

        response = SimulationResponse(
            status=SimulationStatus.COMPLETED,
            simulation_id=sim_id,
            anomaly_id=request.anomaly_id,
            scenario_type=request.scenario_type,
            horizon_days=request.horizon_days,
            model_name=model_name,
            anomaly_context=anomaly_context,
            assumptions=assumptions,
            limitations=limitations,
            baseline=baseline_summary,
            scenario=scenario_summary,
            delta=delta_summary,
            daily_results=daily_results,
            confidence=confidence,
            source="deterministic",
            explanation=None,
            elasticity_model_version=getattr(elasticity_model, "model_version", None) if elasticity_model else None,
            price_elasticity=getattr(elasticity_model, "price_elasticity", None) if elasticity_model else None,
            discount_sensitivity=getattr(elasticity_model, "discount_sensitivity", None) if elasticity_model else None,
        )

        # 7. Validate 10 Invariants
        validate_simulation_result(response)

        # 8. Narrative Explanation Generation
        if request.include_explanation:
            deterministic_exp = self._generate_deterministic_explanation(response, unit_price, scenario_unit_price)
            if self.llm_provider and isinstance(self.llm_provider, GeminiProvider) and self.llm_provider.api_key:
                try:
                    evidence_pkg = build_simulation_evidence(response)
                    gemini_exp = self.llm_provider.generate_simulation_explanation(evidence_pkg)
                    response.explanation = gemini_exp
                    response.source = "gemini_explanation"
                except Exception as exc:
                    logger.warning("Gemini explanation failed (%s), falling back to deterministic: %s", type(exc).__name__, exc)
                    response.explanation = deterministic_exp
                    response.source = "deterministic"
            else:
                response.explanation = deterministic_exp
                response.source = "deterministic"

        # 9. Cache and Return
        self._cache[sim_id] = response
        return response
