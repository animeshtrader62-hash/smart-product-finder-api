from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import json
import os
import httpx

# EarnKaro API Configuration
EARNKARO_API = {
    'url': 'https://ekaro-api.affiliaters.in/api/converter/public',
    'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2OTM0MDg5MzlkOTM5ZWQyMDI5YTZhZTkiLCJlYXJua2FybyI6IjQ3MTI3OTAiLCJpYXQiOjE3NjUwMTgzMDR9.CV4yf6iQ2IZ2RHv8FIWbzHROu4bnV1RRvgTa_JmvSFI'
}

class ConvertRequest(BaseModel):
    url: str

app = FastAPI(
    title="Smart Product Finder API",
    description="API for finding deals from Flipkart, Myntra, Amazon with affiliate links",
    version="1.0.0"
)

# Allow frontend (Netlify/GitHub Pages) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load products from JSON file
def load_products():
    json_path = os.path.join(os.path.dirname(__file__), "products.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

PRODUCTS = load_products()


@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Smart Product Finder API is working!",
        "total_products": len(PRODUCTS),
        "endpoints": {
            "search": "/search",
            "categories": "/categories",
            "platforms": "/platforms",
            "docs": "/docs"
        }
    }


@app.get("/search")
def search_products(
    platform: Optional[str] = Query(None, description="Platform: Flipkart, Myntra, Amazon"),
    category: Optional[str] = Query(None, description="Category: Shoes, T-Shirt, Electronics, etc."),
    brand: Optional[str] = Query(None, description="Brand name"),
    min_price: Optional[int] = Query(None, description="Minimum price", ge=0),
    max_price: Optional[int] = Query(None, description="Maximum price", ge=0),
    min_discount: Optional[int] = Query(None, description="Minimum discount percentage", ge=0, le=100),
    min_rating: Optional[float] = Query(None, description="Minimum rating", ge=0, le=5),
    sort_by: Optional[str] = Query(None, description="Sort by: price_low, price_high, discount, rating"),
    limit: Optional[int] = Query(50, description="Maximum results to return", ge=1, le=100)
):
    """
    Search products with multiple filters.
    Returns matching products with affiliate links.
    """
    results = PRODUCTS.copy()

    # Apply filters
    if platform:
        results = [p for p in results if p["platform"].lower() == platform.lower()]

    if category:
        results = [p for p in results if category.lower() in p["category"].lower()]

    if brand:
        results = [p for p in results if brand.lower() in p["brand"].lower()]

    if min_price is not None:
        results = [p for p in results if p["price"] >= min_price]

    if max_price is not None:
        results = [p for p in results if p["price"] <= max_price]

    if min_discount is not None:
        results = [p for p in results if p["discount"] >= min_discount]

    if min_rating is not None:
        results = [p for p in results if p["rating"] >= min_rating]

    # Apply sorting
    if sort_by:
        if sort_by == "price_low":
            results.sort(key=lambda x: x["price"])
        elif sort_by == "price_high":
            results.sort(key=lambda x: x["price"], reverse=True)
        elif sort_by == "discount":
            results.sort(key=lambda x: x["discount"], reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda x: x["rating"], reverse=True)

    # Apply limit
    results = results[:limit]

    return {
        "success": True,
        "count": len(results),
        "filters_applied": {
            "platform": platform,
            "category": category,
            "brand": brand,
            "min_price": min_price,
            "max_price": max_price,
            "min_discount": min_discount,
            "min_rating": min_rating,
            "sort_by": sort_by
        },
        "products": results
    }


@app.get("/categories")
def get_categories():
    """Get all available categories"""
    categories = list(set(p["category"] for p in PRODUCTS))
    categories.sort()
    return {"categories": categories, "count": len(categories)}


@app.get("/platforms")
def get_platforms():
    """Get all available platforms"""
    platforms = list(set(p["platform"] for p in PRODUCTS))
    platforms.sort()
    return {"platforms": platforms, "count": len(platforms)}


@app.get("/brands")
def get_brands(category: Optional[str] = None):
    """Get all available brands, optionally filtered by category"""
    filtered = PRODUCTS
    if category:
        filtered = [p for p in PRODUCTS if category.lower() in p["category"].lower()]
    
    brands = list(set(p["brand"] for p in filtered))
    brands.sort()
    return {"brands": brands, "count": len(brands)}


@app.get("/deals")
def get_top_deals(limit: int = Query(10, ge=1, le=50)):
    """Get top deals sorted by discount percentage"""
    deals = sorted(PRODUCTS, key=lambda x: x["discount"], reverse=True)[:limit]
    return {
        "success": True,
        "count": len(deals),
        "deals": deals
    }


@app.get("/product/{product_id}")
def get_product(product_id: int):
    """Get a specific product by ID"""
    for product in PRODUCTS:
        if product["id"] == product_id:
            return {"success": True, "product": product}
    return {"success": False, "message": "Product not found"}


@app.post("/convert")
async def convert_to_affiliate(request: ConvertRequest):
    """Convert any URL to EarnKaro affiliate link"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                'Authorization': f'Bearer {EARNKARO_API["token"]}',
                'Content-Type': 'application/json'
            }
            payload = {
                'deal': request.url,
                'convert_option': 'convert_only'
            }
            
            response = await client.post(EARNKARO_API['url'], headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') == 1:
                    return {
                        "success": True,
                        "affiliate_url": data.get('data', request.url),
                        "original_url": request.url
                    }
                else:
                    return {
                        "success": False,
                        "message": data.get('message', 'Conversion failed'),
                        "original_url": request.url
                    }
            else:
                return {
                    "success": False,
                    "message": f"API error: {response.status_code}",
                    "original_url": request.url
                }
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "original_url": request.url
        }


@app.get("/convert")
async def convert_to_affiliate_get(url: str = Query(..., description="URL to convert")):
    """Convert any URL to EarnKaro affiliate link (GET version)"""
    request = ConvertRequest(url=url)
    return await convert_to_affiliate(request)


# For running with: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
