from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Inicializar a FastAPI
app = FastAPI(title="Classificador de Texto com FastAPI")

# Carregar o modelo e tokenizer
model_name = "Lorero/bert-treinado-pedidos-completo"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Modelo de dados para entrada
class TextInput(BaseModel):
    text: str

# Rota principal para classificação
@app.post("/classify")
async def classify_text(input_data: TextInput):
    try:
        # Tokenizar o texto de entrada
        inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Fazer a previsão
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extrair rótulos e scores
        labels = model.config.id2label  # Obtém o mapeamento de IDs para rótulos
        result = {labels[i]: float(probabilities[0][i]) for i in range(len(labels))}
        return {"success": True, "predictions": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Rota de saúde (opcional)
@app.get("/")
def health_check():
    return {"message": "API está funcionando!"}
