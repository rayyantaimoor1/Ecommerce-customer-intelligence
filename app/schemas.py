from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    recency: int = Field(..., ge=0, description="Days since the customer's last purchase")
    frequency: int = Field(..., ge=0, description="Number of distinct orders placed")
    monetary: float = Field(..., ge=0, description="Total amount spent across all orders")
    avg_basket_value: float = Field(..., ge=0, description="Average spend per order")
    avg_items_per_basket: float = Field(..., ge=0, description="Average number of items per order")
    unique_products: int = Field(..., ge=0, description="Number of distinct products purchased")
    cancellation_rate: float = Field(..., ge=0, le=1, description="Share of orders that were cancelled (0-1)")
    customer_lifespan_days: int = Field(..., ge=0, description="Days between first and most recent order")

    class Config:
        json_schema_extra = {
            "example": {
                "recency": 12,
                "frequency": 8,
                "monetary": 650.0,
                "avg_basket_value": 81.25,
                "avg_items_per_basket": 6.5,
                "unique_products": 22,
                "cancellation_rate": 0.05,
                "customer_lifespan_days": 210
            }
        }


class ClusterPrediction(BaseModel):
    cluster_id: int
    persona: str
    input_echo: CustomerInput


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
