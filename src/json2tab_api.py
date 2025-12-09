# json2tab_api.py

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import json2tab  # your main script

app = FastAPI(
    title="json2tab API",
    description="API to convert Songsterr URLs to ASCII tablature with SQLite cache.",
    version="1.0.0",
)


class TabResponse(BaseModel):
    url: str
    tab: str


@app.get("/tab", response_model=TabResponse)
def get_tab(
    url: str = Query(..., description="Songsterr URL"),
    wrap: bool = Query(False, description="Apply wrap by measures"),
    width: Optional[int] = Query(
        None,
        description="Max width in characters (if wrap=true). If not indicated, it is not limited.",
    ),
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

    return TabResponse(url=url, tab=tab_text)
