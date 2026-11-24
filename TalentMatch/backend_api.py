"""
TalentMatch AI - Backend Empresarial de Referencia (FastAPI + PyTorch + SQLite)
Ejecución local:
  pip install -r requirements.txt
  uvicorn backend_api:app --reload --port 8000
"""
import hashlib
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

app = FastAPI(title="TalentMatch AI V4 API", version="4.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "talentmatch_audit.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            hv_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            current_revision INTEGER NOT NULL,
            dataset_version TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS offers (
            offer_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_runs (
            search_id TEXT PRIMARY KEY,
            algo_version TEXT NOT NULL,
            epochs INTEGER NOT NULL,
            hv_ds TEXT NOT NULL,
            of_ds TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


class OptimizedSemanticAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class MatchRequest(BaseModel):
    candidates: List[Dict[str, Any]]
    offers: List[Dict[str, Any]]
    algo_version: str = "V4"
    epochs: int = 15
    top_n: int = 500


@app.get("/health")
def health():
    return {"status": "ok", "engine": "PyTorch-Autoencoder-256D", "seed": SEED}


@app.post("/api/v4/match")
def run_matching(req: MatchRequest):
    if not req.candidates or not req.offers:
        raise HTTPException(status_code=400, detail="Se requieren candidatos y ofertas activas.")

    cand_docs = [
        f"{c.get('Título Profesional','')} " * 3
        + f"{c.get('Habilidades','')} " * 2
        + f"{c.get('Perfil Profesional','')} {c.get('Experiencia Laboral','')}"
        for c in req.candidates
    ]
    offer_docs = [
        f"{o.get('Título de la Oferta','')} " * 3
        + f"{o.get('Requisitos y Habilidades','')} " * 2
        + f"{o.get('Descripción del Cargo','')} {o.get('Área / Sector','')}"
        for o in req.offers
    ]

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=3500)
    X = vec.fit_transform(cand_docs + offer_docs).toarray().astype(np.float32)

    tensor_x = torch.tensor(X)
    model = OptimizedSemanticAutoencoder(X.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=req.epochs)
    criterion = nn.MSELoss()

    model.train()
    loss_history = []
    for ep in range(1, req.epochs + 1):
        optimizer.zero_grad()
        recon, _ = model(tensor_x)
        loss = criterion(recon, tensor_x)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append({"epoch": ep, "loss": round(float(loss.item()), 6)})

    model.eval()
    with torch.no_grad():
        _, Z = model(tensor_x)
        Z_np = Z.numpy()

    n_c = len(req.candidates)
    sim_dl = cosine_similarity(Z_np[:n_c], Z_np[n_c:])
    sim_lex = cosine_similarity(X[:n_c], X[n_c:])

    rankings = {}
    for i, c in enumerate(req.candidates):
        hv_id = c.get("hvId", f"HV-{i+1:04d}")
        scores = 0.55 * sim_dl[i] + 0.45 * sim_lex[i]
        order = np.argsort(-scores)[: req.top_n]
        rankings[hv_id] = [
            {
                "rank": idx + 1,
                "offerId": req.offers[j].get("offerId", f"OF-{j+1:05d}"),
                "rawScore": round(float(scores[j]) * 100, 2)
            }
            for idx, j in enumerate(order)
        ]

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "algoVersion": req.algo_version,
        "lossHistory": loss_history,
        "rankings": rankings
    }
