import torch
import pickle
import io
from fastapi import FastAPI, Response
from PIL import Image
from pydantic import BaseModel

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pickle

with open("my_custom_model.pkl", "rb") as f:
    model = pickle.load(f)

model = model.to(device)
model.eval()

# Correct loading for pickle models saved on GPU
# model = torch.load(
#     "my_custom_model.pkl",
#     map_location=torch.device("cpu"),
#     pickle_module=pickle,
#     weights_only=False
# )

# model = model.to(device)
# model.eval()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(request: PromptRequest):

    with torch.no_grad():

        img = Image.new("RGB", (512,512), color=(73,109,137))

        byte_io = io.BytesIO()
        img.save(byte_io, "PNG")

        return Response(content=byte_io.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)