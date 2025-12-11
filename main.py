from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
import json
import os
import httpx
import urllib.parse
import re
import time
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > minute_ago]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        self.requests[client_ip].append(now)
        return True
    
    def cleanup(self):
        """Remove old entries to prevent memory leak"""
        now = time.time()
        minute_ago = now - 60
        to_delete = [ip for ip, times in self.requests.items() if all(t < minute_ago for t in times)]
        for ip in to_delete:
            del self.requests[ip]

rate_limiter = RateLimiter(requests_per_minute=60)

# EarnKaro API Configuration
EARNKARO_TOKEN = os.getenv("EARNKARO_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2OTM0MDg5MzlkOTM5ZWQyMDI5YTZhZTkiLCJlYXJua2FybyI6IjQ3MTI3OTAiLCJpYXQiOjE3NjUwMTgzMDR9.CV4yf6iQ2IZ2RHv8FIWbzHROu4bnV1RRvgTa_JmvSFI")
EARNKARO_API_URL = "https://ekaro.in/api/generateLink"
FK_AFFILIATE_ID = os.getenv("FK_AFFILIATE_ID", "desideals")
FK_AFFEXT_ID = os.getenv("FK_AFFEXT_ID", "ext1")

class ConvertRequest(BaseModel):
    url: str

class DirectLinkRequest(BaseModel):
    store: str  # flipkart or myntra
    query: str
    brand: Optional[str] = None
    price_min: Optional[int] = 0
    price_max: Optional[int] = 999999
    discount: Optional[int] = 0
    color: Optional[str] = None

# ═══════════════════════════════════════════════════════════════
# URL BUILDERS (Same as Telegram Bot)
# ═══════════════════════════════════════════════════════════════
def build_flipkart_url(query: str, brand: str = None, price_min: int = 0, 
                       price_max: int = 999999, discount: int = 0) -> str:
    """Build Flipkart URL with accurate filters"""
    search_terms = []
    if brand and brand.lower() not in ["all", "all brands", ""]:
        search_terms.append(brand)
    search_terms.append(query)
    q = urllib.parse.quote(" ".join(search_terms))
    
    url = f"https://www.flipkart.com/search?q={q}"
    
    if price_max < 999999:
        url += f"&p%5B%5D=facets.price_range.from%3D{price_min}"
        url += f"&p%5B%5D=facets.price_range.to%3D{price_max}"
    
    if discount:
        url += f"&p%5B%5D=facets.discount_range%5B%5D%3D{discount}%25+or+more"
    
    url += "&sort=popularity"
    return url

def build_myntra_url(search: str, brand: str = None, price_min: int = 0,
                     price_max: int = 999999, discount: int = 0, color: str = None) -> str:
    """Build Myntra URL with accurate filters"""
    search_path = search.lower().replace(" ", "-").replace("'", "")
    url = f"https://www.myntra.com/{search_path}"
    
    params = []
    
    if brand and brand.lower() not in ["all", "all brands", ""]:
        params.append(f"f=Brand%3A{urllib.parse.quote(brand)}")
    
    if price_max < 999999:
        params.append(f"price={price_min}%2C{price_max}")
    
    if discount:
        params.append(f"discount={discount}%3A100")
    
    if color and color.lower() != "any":
        params.append(f"f=Color%3A{urllib.parse.quote(color)}")
    
    params.append("sort=popularity")
    
    if params:
        url += "?" + "&".join(params)
    
    return url

def build_ajio_url(query: str, brand: str = None, price_min: int = 0,
                   price_max: int = 999999, discount: int = 0) -> str:
    """Build Ajio URL with filters - simplified to avoid blocks"""
    # Ajio blocks complex URLs, use simple search
    search_query = query
    if brand and brand.lower() not in ["all", "all brands", ""]:
        search_query = f"{brand} {query}"
    
    # Use simple search URL that Ajio accepts
    url = f"https://www.ajio.com/search/?text={urllib.parse.quote(search_query)}"
    
    return url

async def to_affiliate(url: str) -> str:
    """Convert URL to affiliate link using EarnKaro API (same as bot)"""
    
    # Method 1: EarnKaro API (primary - works for all stores)
    if EARNKARO_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    "https://ekaro.in/api/generateLink",
                    json={"originalLink": url},
                    headers={
                        "Authorization": f"Bearer {EARNKARO_TOKEN}",
                        "Content-Type": "application/json"
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Check different response formats
                    converted = (
                        data.get("data", {}).get("convertedLink") or
                        data.get("data", {}).get("link") or
                        data.get("convertedLink") or
                        data.get("link")
                    )
                    if converted:
                        return converted
        except Exception as e:
            print(f"EarnKaro API error: {e}")
    
    # Method 2: Flipkart Direct Affiliate (fallback for Flipkart)
    if "flipkart.com" in url and FK_AFFILIATE_ID:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}affid={FK_AFFILIATE_ID}&affExtParam1={FK_AFFEXT_ID}"
    
    # Fallback: Return original URL with tracking
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}utm_source=desideals_web"

app = FastAPI(
    title="Smart Product Finder API",
    description="API for finding deals from Flipkart, Myntra, Amazon with affiliate links",
    version="1.0.0"
)

# Allowed origins for CORS (more secure than "*")
ALLOWED_ORIGINS = [
    "https://animeshtrader62-hash.github.io",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "https://smart-deal-finder.netlify.app",  # If using Netlify
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"success": False, "message": "Too many requests. Please wait."}
        )
    
    response = await call_next(request)
    return response

# Load products from JSON file (cached)
_products_cache = None
_products_loaded_time = 0

def load_products():
    global _products_cache, _products_loaded_time
    
    # Cache for 10 minutes
    if _products_cache and (time.time() - _products_loaded_time) < 600:
        return _products_cache
    
    json_path = os.path.join(os.path.dirname(__file__), "products.json")
    with open(json_path, "r", encoding="utf-8") as f:
        _products_cache = json.load(f)
        _products_loaded_time = time.time()
    return _products_cache

# Initial load
PRODUCTS = load_products()


# ═══════════════════════════════════════════════════════════════
# INPUT SANITIZATION
# ═══════════════════════════════════════════════════════════════
def sanitize_input(text: str, max_length: int = 200) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not text:
        return ""
    # Remove dangerous characters and limit length
    text = re.sub(r'[<>\"\\;{}|]', '', text)
    text = text.strip()[:max_length]
    return text


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
    q: Optional[str] = Query(None, description="Full text search query", max_length=200),
    platform: Optional[str] = Query(None, description="Platform: Flipkart, Myntra, Ajio"),
    category: Optional[str] = Query(None, description="Category: Smartphones, Laptops, Shoes, etc.", max_length=100),
    brand: Optional[str] = Query(None, description="Brand name", max_length=100),
    min_price: Optional[int] = Query(None, description="Minimum price", ge=0),
    max_price: Optional[int] = Query(None, description="Maximum price", ge=0),
    min_discount: Optional[int] = Query(None, description="Minimum discount percentage", ge=0, le=100),
    min_rating: Optional[float] = Query(None, description="Minimum rating", ge=0, le=5),
    sort_by: Optional[str] = Query(None, description="Sort by: price_low, price_high, discount, rating"),
    limit: Optional[int] = Query(50, description="Maximum results to return", ge=1, le=100)
):
    """
    Search products with multiple filters and full-text search.
    Returns matching products with affiliate links.
    """
    # Refresh products cache if needed
    products = load_products()
    results = products.copy()
    
    # Sanitize inputs
    q = sanitize_input(q) if q else None
    platform = sanitize_input(platform, 20) if platform else None
    category = sanitize_input(category, 100) if category else None
    brand = sanitize_input(brand, 100) if brand else None
    
    # Validate sort_by
    valid_sorts = ["price_low", "price_high", "discount", "rating", None]
    if sort_by not in valid_sorts:
        sort_by = None
    
    # Full-text search across title, brand, category, description
    if q:
        q_lower = q.lower()
        results = [p for p in results if 
            q_lower in p.get("title", "").lower() or
            q_lower in p.get("brand", "").lower() or
            q_lower in p.get("category", "").lower() or
            q_lower in p.get("description", "").lower()
        ]

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
            "q": q,
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
        affiliate_url = await to_affiliate(request.url)
        return {
            "success": True,
            "affiliate_url": affiliate_url,
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


@app.post("/generate-link")
async def generate_direct_link(request: DirectLinkRequest):
    """Generate direct product link with filters (same as bot)"""
    try:
        store = request.store.lower()
        
        if store == "myntra":
            url = build_myntra_url(
                search=request.query,
                brand=request.brand,
                price_min=request.price_min,
                price_max=request.price_max,
                discount=request.discount,
                color=request.color
            )
        elif store == "ajio":
            url = build_ajio_url(
                query=request.query,
                brand=request.brand,
                price_min=request.price_min,
                price_max=request.price_max,
                discount=request.discount
            )
        else:  # flipkart (default)
            url = build_flipkart_url(
                query=request.query,
                brand=request.brand,
                price_min=request.price_min,
                price_max=request.price_max,
                discount=request.discount
            )
        
        # Convert to affiliate
        affiliate_url = await to_affiliate(url)
        
        return {
            "success": True,
            "original_url": url,
            "affiliate_url": affiliate_url,
            "store": request.store
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


@app.get("/generate-link")
async def generate_direct_link_get(
    store: str = Query(..., description="Store: flipkart or myntra"),
    query: str = Query(..., description="Search query"),
    brand: Optional[str] = Query(None, description="Brand filter"),
    price_min: Optional[int] = Query(0, description="Minimum price"),
    price_max: Optional[int] = Query(999999, description="Maximum price"),
    discount: Optional[int] = Query(0, description="Minimum discount"),
    color: Optional[str] = Query(None, description="Color filter (Myntra only)")
):
    """Generate direct product link with filters - GET version"""
    request = DirectLinkRequest(
        store=store,
        query=query,
        brand=brand,
        price_min=price_min,
        price_max=price_max,
        discount=discount,
        color=color
    )
    return await generate_direct_link(request)


# For running with: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
