# Smart Product Finder API

A FastAPI backend for searching deals from Flipkart, Myntra & Amazon.

## Endpoints

- `GET /` - Health check
- `GET /search` - Search products with filters
- `GET /deals` - Get top deals
- `GET /categories` - List all categories
- `GET /platforms` - List all platforms
- `GET /brands` - List all brands

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Deployment

Deployed on Render.com
