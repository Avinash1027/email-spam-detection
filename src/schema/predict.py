from pydantic import BaseModel, Field
class Input(BaseModel):
    text: str = Field(..., description="Enter you email content", examples=['Hii, I am John'])

class Predict(BaseModel):
    text: str
    confidence_score: float
    prediction: str