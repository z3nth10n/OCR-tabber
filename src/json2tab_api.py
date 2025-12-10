# json2tab_api.py

from typing import Optional
import os
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel

import json2tab
import requests

app = FastAPI(
    title="json2tab API",
    description="API to convert Songsterr URLs to ASCII tablature with SQLite cache.",
    version="1.0.0",
)

# --- Domain restriction (optional, via ALLOWED_DOMAINS env var) ---

# ALLOWED_DOMAINS puede ser, por ejemplo:
#   ALLOWED_DOMAINS="https://z3nth10n.github.io,https://otro-dominio.com"
_ALLOWED_DOMAINS_ENV = os.getenv("ALLOWED_DOMAINS")

if _ALLOWED_DOMAINS_ENV:
    ALLOWED_DOMAINS = {
        d.strip().rstrip("/")
        for d in _ALLOWED_DOMAINS_ENV.split(",")
        if d.strip()
    }
else:
    ALLOWED_DOMAINS = None  # Sin restricciones


# CORS: si hay dominios permitidos, sólo esos; si no, todo (*)
if ALLOWED_DOMAINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_DOMAINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _extract_origin_domain(request: Request) -> Optional[str]:
    """
    Devuelve 'scheme://host' a partir de Origin o Referer, o None si no hay.
    """
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return None
    parsed = urlparse(origin)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


async def verify_origin(request: Request):
    """
    Si ALLOWED_DOMAINS está definido, sólo acepta peticiones cuyo Origin/Referer
    pertenezca a esa lista. Si no, devuelve 403.
    Si ALLOWED_DOMAINS no está definido, no hace nada.
    """
    if not ALLOWED_DOMAINS:
        # No hay restricción configurada, aceptamos cualquier origen.
        return

    domain = _extract_origin_domain(request)
    if not domain:
        # No viene Origin ni Referer -> la tiramos igual
        raise HTTPException(
            status_code=403,
            detail="Forbidden: missing or invalid Origin/Referer",
        )

    if domain.rstrip("/") not in ALLOWED_DOMAINS:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden origin: {domain}",
        )

class TabResponse(BaseModel):
    url: str
    tab: str

@app.get("/tab", dependencies=[Depends(verify_origin)])
def get_tab(
    url: str = Query(..., description="Songsterr URL"),
    wrap: bool = Query(False, description="Apply wrap by measures"),
    width: Optional[int] = Query(
        None,
        description="Max width in characters (if wrap=true). If not indicated, it is not limited.",
    ),
    txt: bool = Query(False, description="Return plain text response"),
):
    """
    Returns the ASCII tablature for the indicated Songsterr URL.
    Uses the same SQLite cache as the json2tab.py CLI.
    """
    max_width: Optional[int] = None
    if wrap:
        max_width = width if width is not None else None

    try:
        tab_text = json2tab.generate_tab_from_url(
            url,
            max_width=max_width,
            use_cache=True,  # always use cache in the API
        )
    except ValueError as e:
        # For example: no guitars found at that URL
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating tablature: {e}",
        )

    if txt:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Tablature</title>
            <style>
                body {{
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    margin: 0;
                    padding: 20px;
                }}
                pre {{
                    font-family: "Consolas", "Courier New", monospace;
                    font-size: 14px;
                    white-space: pre; /* Ensures no wrapping */
                }}
            </style>
        </head>
        <body>
            <pre>{tab_text}</pre>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    return TabResponse(url=url, tab=tab_text)

@app.get(
    "/songsterr-search",
    dependencies=[Depends(verify_origin)]
)
def songsterr_search(
    pattern: str = Query(..., description="Songsterr search pattern"),
    size: int = Query(10, description="Max results"),
):
    """
    Proxy de búsqueda hacia la API pública de Songsterr.
    Evita problemas de CORS llamándola desde el servidor.
    """
    api_url = f"https://www.songsterr.com/api/songs?size={size}&pattern={pattern}"
    try:
        resp = requests.get(api_url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error calling Songsterr API: {e}",
        )

    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Invalid JSON from Songsterr",
        )

    return data