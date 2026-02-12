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
    model.eval()
    with torch.inference_mode():
        output_logits = model(tensor)
        output=torch.sigmoid(output_logits)
        print("Processed text:", text)
        print("Non-zero features:", text_tf.nnz)
        print("Sample vocab words:", list(tfidf.vocabulary_.keys())[:20])
        confidence = float(output.item())
        prediction = "spam" if confidence > 0.7 else "not spam"
    return {
        "text": text,
        "prediction": prediction,
        "confidence_score" : confidence
        }