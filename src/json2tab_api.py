# json2tab_api.py

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import json2tab  # tu script principal

app = FastAPI(
    title="json2tab API",
    description="API para convertir URLs de Songsterr en tablaturas ASCII con caché SQLite.",
    version="1.0.0",
)


class TabResponse(BaseModel):
    url: str
    tab: str


@app.get("/tab", response_model=TabResponse)
def get_tab(
    url: str = Query(..., description="URL de Songsterr"),
    wrap: bool = Query(False, description="Aplicar wrap por compases"),
    width: Optional[int] = Query(
        None,
        description="Ancho máximo en caracteres (si wrap=true). Si no se indica, no se limita.",
    ),
):
    """
    Devuelve la tablatura ASCII para la URL de Songsterr indicada.
    Usa la misma caché SQLite que el CLI de json2tab.py.
    """
    max_width: Optional[int] = None
    if wrap:
        max_width = width if width is not None else None

    try:
        tab_text = json2tab.generate_tab_from_url(
            url,
            max_width=max_width,
            use_cache=True,  # siempre usamos caché en la API
        )
    except ValueError as e:
        # Por ejemplo: no se encontraron guitarras en esa URL
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando la tablatura: {e}",
        )

    return TabResponse(url=url, tab=tab_text)
