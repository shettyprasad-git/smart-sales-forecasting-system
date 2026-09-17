from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.dependencies import get_current_user
from backend.app.schemas.simulations import SimulationRequest, SimulationResponse
from backend.app.services.simulation_service import SimulationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/simulations", tags=["What-If Simulations"])

simulation_service = SimulationService()


@router.post(
    "",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Run what-if scenario simulation",
    description=(
        "Executes a what-if sales simulation over 7, 30, or 90 days. "
        "Supports demand multipliers, temporary shocks, persistent shifts, trend continuation, "
        "and promotional/holiday scenarios. Unsupported price/discount changes return status requires_model."
    ),
)
def create_simulation(
    request: SimulationRequest,
    current_user=Depends(get_current_user),
) -> SimulationResponse:
    try:
        return simulation_service.run_simulation(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error executing simulation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute simulation: {exc}",
        ) from exc


@router.get(
    "/{simulation_id}",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get cached simulation result",
    description="Retrieves a previously computed simulation result by unique ID.",
)
def get_simulation(
    simulation_id: str,
    current_user=Depends(get_current_user),
) -> SimulationResponse:
    result = simulation_service.get_simulation(simulation_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation with ID '{simulation_id}' not found.",
        )
    return result
