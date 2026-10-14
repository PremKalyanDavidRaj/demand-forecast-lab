"""HTTP API for historical bike-demand forecasts."""

from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.service import ForecastService


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forecast_date: date


class HourlyForecast(BaseModel):
    timestamp: datetime
    prediction: float = Field(ge=0)
    lower: float = Field(ge=0)
    upper: float = Field(ge=0)
    prediction_source: Literal[
        "same_hour_last_week",
        "same_hour_yesterday",
        "training_median",
    ]


class ForecastResponse(BaseModel):
    forecast_date: date
    model_type: str
    model_sha256: str
    nominal_interval_coverage: float = Field(gt=0, lt=1)
    forecast: list[HourlyForecast] = Field(min_length=24, max_length=24)
    note: str


class HealthResponse(BaseModel):
    status: str
    model_type: str
    first_forecast_date: date
    last_forecast_date: date


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ForecastService()
    try:
        yield
    finally:
        app.state.service = None


app = FastAPI(
    title="Demand Forecast Lab",
    version="0.1.0",
    description=(
        "Historical bike-rental forecasts using a validation-selected "
        "seasonal baseline and empirical hourly prediction intervals. "
        "This is not a live demand feed."
    ),
    lifespan=lifespan,
)


def get_service(request: Request):
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "Forecast service is not ready.")
    return service


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    service = get_service(request)
    return {
        "status": "ok",
        "model_type": service.artifact["model_type"],
        "first_forecast_date": service.first_date.date(),
        "last_forecast_date": service.last_date.date(),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(body: ForecastRequest, request: Request):
    service = get_service(request)

    if not (
        service.first_date.date()
        <= body.forecast_date
        <= service.last_date.date()
    ):
        raise HTTPException(
            422,
            f"Choose a date from {service.first_date.date()} "
            f"through {service.last_date.date()}.",
        )

    return service.forecast(body.forecast_date)


@app.get("/demo", include_in_schema=False)
def demo():
    from pathlib import Path
    from fastapi.responses import FileResponse

    page = Path(__file__).resolve().parents[1] / "static" / "index.html"
    return FileResponse(page)
