from fastapi import FastAPI, File, UploadFile
import os
from car_check import prepare_img_224, car_categories_check
from damage_check import prepare_flat, car_damage_check

app = FastAPI()
UPLOAD_FOLDER = "testimages"
@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:
        # Create the specified location if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        # Get the file name
        filename = file.filename
        # Create the file path
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        # Save the file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"file_path": file_path}
    except Exception as e:
        return {"error": str(e)}

@app.post("/carcheck/")
async def upload_image(file: UploadFile):
    try:
        # Create the specified location if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        # Get the file name
        filename = file.filename
        # Create the file path
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        # Save the file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        img_224 = prepare_img_224(file_path)
        check = car_categories_check(img_224)

        if check is False:
            return {"result": "Not a car"}
        else:
            return {"result": "Car"}        
    except Exception as e:
        return {"error": str(e)}

@app.post("/damagecheck/")
async def upload_image(file: UploadFile):
    try:
        # Create the specified location if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        # Get the file name
        filename = file.filename
        # Create the file path
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        # Save the file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        img_224 = prepare_img_224(file_path)
        img_flat = prepare_flat(img_224)
        check = car_damage_check(img_flat)

        if check is False:
            return {"result": "Not damaged"}
        else:
            return {"result": "Damaged"}        
    except Exception as e:
        return {"error": str(e)}