cdfrom subprocess import Popen
import uvicorn
import base64
import re
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Path, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from analysis import classification_final, segmentation_final

"""
The following is the main code for creating and mainting the server-side logic for handling HTTP requests

Some part of the following implementation have been partically inspired by works outlined in the following links:
    1. https://linuxtut.com/en/bd6c5c486b31e87db44b/
    2. https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-with-fastapi-docker-and-github-actions-13374cbd638a
"""

# Initializing the FastAPI framework
app = FastAPI()

# Adapted CORSMiddleware Implementation 
# from https://fastapi.tiangolo.com/tutorial/cors/
# This is done so that the locally hosted server could handle
# locally hosted client requests
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://cosed-net-client.herokuapp.com/",
    "https://cosednetwork.com/",
    "https://www.cosednetwork.com/",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Creating the class structure for the different client requests and server responses
# The following class represents the client requests
class RetinalImage(BaseModel):
    img: str
# The following class represents the server responses
class ResultsOut(BaseModel):
    classification: str
    img_HE: str
    img_MA: str
    img_SE: str
    img_EX: str


def to_base64(image):
    """
    This function converts the image into a base64 string 
    Args:
        image (PIL.Image): The image to be converted into a base64 string
    Returns:
        str: The base64 string of the image
    """
    base64_image = image.copy()
    with BytesIO() as buf:
        base64_image.save(buf, 'jpeg')
        image_bytes = buf.getvalue()
    img_encoded = base64.b64encode(image_bytes)
    return img_encoded

@app.post("/upload", response_model=ResultsOut)
async def receive_retinal_image(client_data: RetinalImage):
    """
    This function receives the client request and processes it by 
    first converting the base64 string into a PIL image and then 
    calling the analysis functions to perform the classification and segmentation tasks on the specified image
    """
    image_data = client_data.img
    image_base64 = re.sub('^data:image/.+;base64,', '', image_data)
    if(image_base64):
        print("Image Recived!")

    # Convert the base64 string into a PIL image
    img_converted = base64.b64decode(image_base64)
    img_converted = BytesIO(img_converted) 
    img_converted = Image.open(img_converted)

    # Perform the classification and segmentation tasks on the image

    # Classification (more detailed description in the analysis.py file)
    classification_output = np.array2string(classification_final(img_converted))

    # Segmentation (more detailed description in the analysis.py file)
    image_ex, image_he, image_ma, image_se  = segmentation_final(img_converted)

    # Convert the newly segmented images into base64 strings
    img_ex_encoded = to_base64(image_ex)
    img_he_encoded = to_base64(image_he)
    img_ma_encoded = to_base64(image_ma)
    img_se_encoded = to_base64(image_se)

    # Return the server response
    return {"classification": classification_output,
            "img_EX": img_ex_encoded,
            "img_HE": img_he_encoded,
            "img_MA": img_ma_encoded,
            "img_SE": img_se_encoded}


# For testing the request/response 
@app.get("/")
async def get_results():
    return {"hello": "world"}


# This starts the server
if __name__ == '__main__':
    Popen(['python3', '-m', 'https_redirect'])  # Add this
    uvicorn.run(
        'server:app', port=443, host='0.0.0.0',
        # reload=True, reload_dirs=['html_files'],
        ssl_keyfile=r'/home/nvidia/Desktop/3_WebDev/server/SSL/cosednet.key',
        ssl_certfile=r'/home/nvidia/Desktop/3_WebDev/server/SSL/cosednetwork_com_crt.pem',
        ssl_ca_certs=r'/home/nvidia/Desktop/3_WebDev/server/SSL/cosednetwork_com.ca-bundle')