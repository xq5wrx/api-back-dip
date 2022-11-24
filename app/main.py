from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from uvicorn import run
from fastapi.responses import FileResponse
import os
import model

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Welcome to the API Backend for Dip project"}


@app.get("/result")
async def getResult():
    content = "<body> <img src='/resultImage' alt='imgpng'> </body>"
    return HTMLResponse(content=content)


@app.get("/resultImage")
async def getImage():
    return FileResponse("output_graphs/new_graph.png")


@app.get("/model")
async def makeModel():
    result = model.generateModel()
    if result:
        return {"status": "ok", "message": "Model successfully created!"}
    return {"status": "error", "message": "Model not created."}


@app.post("/import")
async def importData(file: UploadFile):
    result = model.importDataSql(file)
    if result:
        return {"status": "ok", "message": "Data successfully imported!"}
    return {"status": "error", "message": "Data not imported."}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)
