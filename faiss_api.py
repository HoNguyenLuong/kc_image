# File: main.py

import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
import json
import io
import cairosvg  # To handle SVG conversion to raster format

# File paths for saving the index and metadata
INDEX_FILE_PATH = "cache/faiss_index.bin"
METADATA_FILE_PATH = "cache/metadata.json"

# Check if a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the CLIP model and processor, and move the model to the GPU if available
model = CLIPModel.from_pretrained("models/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch32")

# Initialize Faiss index using L2 distance for similarity search
embedding_dim = 512  # CLIP model's output dimension for image features
index = faiss.IndexFlatL2(embedding_dim)

# Initialize FastAPI app
app = FastAPI()

# Storage for mapping index positions to image URLs and metadata
image_database: List[str] = []  # List to store image URLs
metadata_database: List[Optional[Dict[str, str]]] = []  # List to store metadata

# Load the index and metadata if they exist
if os.path.exists(INDEX_FILE_PATH):
    index = faiss.read_index(INDEX_FILE_PATH)
    print(f"Loaded Faiss index from {INDEX_FILE_PATH}")

if os.path.exists(METADATA_FILE_PATH):
    with open(METADATA_FILE_PATH, "r") as f:
        data = json.load(f)
        image_database = data["image_database"]
        metadata_database = data["metadata_database"]
    print(f"Loaded metadata from {METADATA_FILE_PATH}")


class InsertRequest(BaseModel):
    url: str
    metadata: Optional[Dict[str, str]] = (
        None  # Metadata is optional and can be a dictionary of strings
    )


class SearchRequest(BaseModel):
    url: str
    k: int = 10  # Number of nearest neighbors to return


def get_image_embedding(url: str):
    """Helper function to generate an embedding for a given image URL, including SVG handling."""
    try:
        # Define headers to mimic a standard browser request
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
            )
        }
        
        # Load the image data from the URL with headers
        response = requests.get(url, stream=True, headers=headers)
        if not response.ok:
            raise HTTPException(
                status_code=400, detail=f"Failed to retrieve the image from the URL: {url}. Status code: {response.status_code}"
            )

        # Determine the content type from the response headers
        content_type = response.headers.get("Content-Type", "")
        
        # Check if SVG, handle SVG files by converting to PNG
        if "svg" in content_type:
            try:
                png_data = cairosvg.svg2png(bytestring=response.content)
                image = Image.open(io.BytesIO(png_data))
            except Exception as svg_error:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process SVG image: {svg_error}"
                )
        else:
            # Handle raster images directly
            try:
                image = Image.open(response.raw)
            except Exception as img_error:
                raise HTTPException(
                    status_code=400, detail=f"Failed to open raster image: {img_error}"
                )

        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process the image to get embeddings
        inputs = processor(images=image, return_tensors="pt", padding=True)

        # Move inputs to the GPU if available
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)

        # Move embeddings back to CPU for compatibility with Faiss
        return embeddings.cpu().numpy()

    except requests.exceptions.RequestException as req_error:
        raise HTTPException(status_code=400, detail=f"Network error: {req_error}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")


def save_index_and_metadata():
    """Helper function to save Faiss index and metadata to disk."""
    faiss.write_index(index, INDEX_FILE_PATH)
    with open(METADATA_FILE_PATH, "w") as f:
        json.dump({"image_database": image_database, "metadata_database": metadata_database}, f)
    print(f"Saved Faiss index to {INDEX_FILE_PATH} and metadata to {METADATA_FILE_PATH}")


@app.post("/insert")
def insert_image(request: InsertRequest):
    """Endpoint to insert an image into the Faiss index along with metadata."""
    # Insert url database Minh
    # title+description

    embedding = get_image_embedding(request.url)
    index.add(embedding)  # Add embedding to the Faiss index
    image_database.append(request.url)  # Store URL for reference
    metadata_database.append(request.metadata)  # Store metadata for reference
    # Save index and metadata to disk
    save_index_and_metadata()

    return {"message": "Image inserted successfully", "index_id": len(image_database) - 1}


@app.post("/search")
def search_image(request: SearchRequest):
    """Endpoint to search for similar images in the Faiss index."""

    # Search using gglen output
    embedding = get_image_embedding(request.url)

    if index.ntotal == 0:
        raise HTTPException(status_code=404, detail="The index is empty. Insert images first.")

    # Search using L2 distance
    distances, indices = index.search(embedding, request.k)

    # Build the search results, including metadata
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append(
            {
                "index_id": int(idx),
                "url": image_database[idx],
                "metadata": metadata_database[idx],
                "distance": float(dist),
            }
        )

    return {"results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9920)
