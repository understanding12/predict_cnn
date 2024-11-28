import torch

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import CNN

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# load model
model = CNN(10)
model.load_state_dict(torch.load('checkpoints/cnn.pth', weights_only=True))
# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    image = torch.tensor(list(map(int, image[1:-1].split(','))), dtype=torch.float32).reshape((1, 1, 28, 28))
    pred = model.predict(image)
    return {'prediction': str(pred)}

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
