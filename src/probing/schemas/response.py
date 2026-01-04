from pydantic import BaseModel, Field

class ClassificationResponse(BaseModel):
    labels: list[int] = Field(..., description="List of predicted labels for each input text")
    texts: list[str] = Field(..., description="List of input texts corresponding to the labels")