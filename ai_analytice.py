# app.py (improved with simple reasoning)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import numpy as np
import os
from pyngrok import ngrok

app = FastAPI()


# ----- Model definition -----
class ImageHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.fc1 = nn.Linear(in_features, 256)
        self.act = nn.ReLU()
        self.dir_head = nn.Linear(256, 1)
        self.reg_head = nn.Linear(256, 1)

    def forward(self, x):
        feat = self.backbone(x)
        h = self.act(self.fc1(feat))
        p = torch.sigmoid(self.dir_head(h)).squeeze(-1)
        pct = self.reg_head(h).squeeze(-1)
        return p, pct


# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageHeadModel()

if os.path.exists("model.pt"):
    model.load_state_dict(torch.load("model.pt", map_location=device))
    print("Loaded trained model.pt")
else:
    print("No model.pt found — using untrained model (random predictions)")

model.to(device)
model.eval()

# preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        p_bull, pct = model(x)
        p_bull = float(p_bull.cpu().numpy())
        pct = float(pct.cpu().numpy())

    # reasoning
    if p_bull > 0.6:
        reason = "Price shows upward momentum / bullish pattern detected"
    elif p_bull < 0.4:
        reason = "Price shows downward momentum / bearish pattern detected"
    else:
        reason = "Price movement is unclear / neutral trend"

    result = {
        "bullish_prob": round(p_bull * 100, 2),
        "bearish_prob": round((1 - p_bull) * 100, 2),
        "expected_move_pct": round(pct, 4),
        "confidence": round(max(p_bull, 1 - p_bull) * 100, 2),
        "reason": reason
    }
    return JSONResponse(content=result)


if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print(" Public URL:", public_url)

    uvicorn.run(app, host="0.0.0.0", port=8000)
