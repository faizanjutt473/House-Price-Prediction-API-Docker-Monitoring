"""Pydantic models = input validation for the API."""
from pydantic import BaseModel, ConfigDict, Field


class HouseFeatures(BaseModel):
    """One census block group. Ranges are generous, but reject impossible values."""

    model_config = ConfigDict(
        extra="forbid",  # unknown fields -> 422
        json_schema_extra={
            "example": {
                "med_inc": 8.3252,
                "house_age": 41,
                "ave_rooms": 6.98,
                "ave_bedrms": 1.02,
                "population": 322,
                "ave_occup": 2.56,
                "latitude": 37.88,
                "longitude": -122.23,
            }
        },
    )

    med_inc: float = Field(..., gt=0, le=20, allow_inf_nan=False, description="Median income (tens of thousands of USD)")
    house_age: float = Field(..., ge=0, le=100, allow_inf_nan=False, description="Median house age in the block (years)")
    ave_rooms: float = Field(..., gt=0, le=200, allow_inf_nan=False, description="Average rooms per household")
    ave_bedrms: float = Field(..., gt=0, le=50, allow_inf_nan=False, description="Average bedrooms per household")
    population: float = Field(..., gt=0, le=50_000, allow_inf_nan=False, description="Block population")
    ave_occup: float = Field(..., gt=0, le=2_000, allow_inf_nan=False, description="Average household members")
    latitude: float = Field(..., ge=32, le=42.5, allow_inf_nan=False, description="Latitude (California)")
    longitude: float = Field(..., ge=-125, le=-114, allow_inf_nan=False, description="Longitude (California)")


class PredictionResponse(BaseModel):
    request_id: str
    predicted_median_house_value: float = Field(..., description="Predicted median house value in USD")
    model_version: str
    inference_ms: float = Field(..., description="Time spent inside model.predict, in milliseconds")
