#!/usr/bin/env python3
"""Wrapper pyfunc de MLflow para el clasificador de lesiones de piel.

Empaqueta el modelo (ResNet18 fine-tuneado) + el preprocesamiento + Grad-CAM
en un único artefacto MLflow, para que el endpoint de Databricks Model
Serving devuelva en una sola llamada tanto las probabilidades por clase
como el mapa de calor Grad-CAM.

Por qué hace falta este wrapper (y no alcanza con mlflow.pytorch.log_model):
un endpoint de serving normal solo expone forward pass. Grad-CAM necesita
un backward pass sobre una capa interna (layer4), así que ese cálculo tiene
que vivir *dentro* del artefacto servido — acá, en predict().

No se usa directamente: lo instancia y loguea scripts/register_model.py.
Cuando el endpoint lo carga, `skin_classifier.py` viaja junto al modelo
como code artifact (ver `code_paths` en register_model.py) para poder
importar CLASS_NAMES/build_model/get_transforms.
"""

import base64
import io

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image


class SkinLesionPyfuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from skin_classifier import CLASS_NAMES, build_model, get_transforms

        self.class_names = CLASS_NAMES
        self.transform = get_transforms(train=False)
        self.model = build_model(pretrained=False)
        state_dict = torch.load(context.artifacts["weights"], map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _predict_one(self, image_b64: str) -> dict:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        x = self.transform(img).unsqueeze(0)

        activations, gradients = {}, {}

        def fwd_hook(module, inp, out):
            activations["v"] = out

        def bwd_hook(module, grad_in, grad_out):
            gradients["v"] = grad_out[0]

        layer = self.model.layer4[-1]
        h1 = layer.register_forward_hook(fwd_hook)
        h2 = layer.register_full_backward_hook(bwd_hook)
        try:
            self.model.zero_grad()
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            class_idx = int(torch.argmax(probs).item())
            logits[0, class_idx].backward()

            acts = activations["v"][0]  # (C, H, W)
            grads = gradients["v"][0]  # (C, H, W)
            weights = grads.mean(dim=(1, 2))
            cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
            cam = cam / (cam.max() + 1e-8)
        finally:
            h1.remove()
            h2.remove()

        return {
            "probs": {self.class_names[i]: float(probs[i].detach()) for i in range(len(self.class_names))},
            # Resolución nativa de layer4 (7x7 para input 224x224), no
            # 224x224 — reduce el payload de la respuesta ~1000x. El
            # cliente la reescala (ver predict_via_endpoint en
            # skin_classifier.py).
            "gradcam": cam.detach().cpu().numpy().tolist(),
        }

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return [self._predict_one(b64) for b64 in model_input["image_b64"]]
