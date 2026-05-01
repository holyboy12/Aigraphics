import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return FileResponse("index.html")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=api_key)

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(data: Prompt):
    if not data.prompt or not data.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=data.prompt,
            size="1024x1024"
        )
        b64 = result.data[0].b64_json
        return {"image": "data:image/png;base64," + b64}
    except Exception:
        logging.exception("Image generation failed")
        raise HTTPException(status_code=500, detail="Image generation failed")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=True)