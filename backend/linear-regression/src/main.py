from fastapi import FastAPI, File, UploadFile
from application.services import TextureAnalysisService

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile):
    """API to accept image and predict class using texture features."""
    try:
        image_bytes = await file.read()
        result = TextureAnalysisService.analyze_texture(image_bytes)
        return result
    except ValueError as e:
        return {"error": str(e)}
