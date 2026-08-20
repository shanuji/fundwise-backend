from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import date, datetime, timedelta
from decimal import Decimal
import casparser
from pyxirr import xirr
import json
import os
import uuid
import httpx
from collections import defaultdict
import traceback

app = FastAPI(title="FundWise Analytics Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "fundwise-backend", "timestamp": datetime.utcnow().isoformat()}

# The remainder of the application is unchanged.
