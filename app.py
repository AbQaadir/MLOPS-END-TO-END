from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Data(BaseModel):
    carat: float
    depth: float
    table: float
    x: float
    y: float
    z: float
    cut: str
    color: str
    clarity: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def get_predict_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict")
async def predict(
    carat: float = Form(...),
    depth: float = Form(...),
    table: float = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    z: float = Form(...),
    cut: str = Form(...),
    color: str = Form(...),
    clarity: str = Form(...)
):
    try:
        data = Data(carat=carat, depth=depth, table=table, x=x, y=y, z=z, cut=cut, color=color, clarity=clarity)
        return JSONResponse(content=data.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
