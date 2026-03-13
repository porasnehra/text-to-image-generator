import io
import torch
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image
from diffusers import StableDiffusionPipeline

app = FastAPI()

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Stable Diffusion model...")

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

print("Model loaded successfully")

class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate_image(request: PromptRequest):

    prompt = request.prompt

    with torch.no_grad():
        image = pipe(prompt).images[0]

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")

    return Response(
        content=img_byte_arr.getvalue(),
        media_type="image/png"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)