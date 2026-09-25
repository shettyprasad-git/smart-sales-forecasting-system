from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from sqlalchemy.orm import Session

from backend.app.core.rate_limiter import rate_limit_heavy_intelligence
from backend.app.database.database import get_db
from backend.app.dependencies import get_current_user
from backend.app.schemas.simulations import SimulationRequest, SimulationResponse
from backend.app.services.dataset_runtime_service import NoActiveDatasetError
from backend.app.services.simulation_service import (
    SimulationService,
    UnsupportedScenarioError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/simulations",
    tags=["What-If Simulations"],
)

simulation_service = SimulationService()


@router.post(
    "",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Run what-if scenario simulation",
    description=(
        "Executes a what-if sales simulation over 7, 30, or 90 days. "
        "Supports demand multipliers, temporary shocks, persistent shifts, trend continuation, "
        "and promotional/holiday scenarios. Unsupported price/discount changes return HTTP 422."
    ),
    dependencies=[Depends(rate_limit_heavy_intelligence)],
)
def create_simulation(
    request: SimulationRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SimulationResponse:
    try:
        return simulation_service.run_simulation(
            request,
            user_id=current_user.id if current_user else None,
            db=db,
        )
    except UnsupportedScenarioError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except NoActiveDatasetError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error executing simulation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute simulation. Internal server error.",
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
