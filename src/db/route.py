from fastapi import APIRouter
import joblib
import torch
from src.schema.predict import Predict
from src.model.model_definition import NNModel
router= APIRouter()
model= NNModel(input_features=5000)
model.load_state_dict(torch.load(r'src\model\model.pth'))
tfidf=joblib.load(r'src\model\tfidf.pkl')
@router.post('/predict', response_model=Predict)
def predict(text: str):
    text_tf= tfidf.transform([text])
    tensor= torch.tensor(text_tf.toarray(), dtype=torch.float32)
    with torch.inference_mode(): 
        output = model(tensor)
        confidence = float(output.item())
        prediction = "spam" if confidence > 0.5 else "not spam"
    return {
        "text": text,
        "prediction": prediction,
        "confidence_score": output
        }