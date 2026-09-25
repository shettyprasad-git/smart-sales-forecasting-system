import logging
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.database import get_db
from backend.app.database.models import User
from backend.app.dependencies import get_current_user_optional
from backend.app.schemas.investigations import InvestigationResponse
from backend.app.services.dataset_runtime_service import NoActiveDatasetError
from backend.app.services.investigation_service import InvestigationService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/investigations",
    tags=["Investigations"],
)

investigation_service = InvestigationService()


@router.get(
    "/{anomaly_id}",
    response_model=InvestigationResponse,
    summary="Investigate sales anomaly root causes",
    description=(
        "Retrieve multi-dimensional root-cause attribution, financial impact quantification, "
        "and evidence decomposition for a specific sales or demand anomaly."
    ),
)
def get_anomaly_investigation(
    anomaly_id: str = Path(
        ...,
        description="Unique anomaly identifier (e.g. anom-20251228-sal-agg)",
    ),
    top_n: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of top product and category contributors to include",
    ),
    include_products: bool = Query(
        default=True,
        description="Whether to include granular product-level contributors",
    ),
    include_categories: bool = Query(
        default=True,
        description="Whether to include category-level contributors",
    ),
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    if settings.is_production and current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        user_id = current_user.id if current_user else None
        return investigation_service.investigate_anomaly(
            anomaly_id=anomaly_id,
            top_n=top_n,
            include_products=include_products,
            include_categories=include_categories,
            user_id=user_id,
            db=db,
        )
    except NoActiveDatasetError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        logger.error("Required dataset file missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Required dataset or file was not found. Internal server error.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Investigation failed for anomaly %s: %s", anomaly_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Investigation failed. Internal server error.",
        ) from exc
