import os
import sys
import urllib.parse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.main import app

async def handler(scope, receive, send):
    if scope.get("type") == "http":
        # Extract path parameter if forwarded by Vercel routes
        query_string = scope.get("query_string", b"").decode("latin-1")
        if query_string:
            params = urllib.parse.parse_qs(query_string, keep_blank_values=True)
            if "path" in params and params["path"]:
                extracted_path = params["path"][0]
                if not extracted_path.startswith("/"):
                    extracted_path = "/" + extracted_path
                scope["path"] = extracted_path
                
        # Handle path fallback
        curr_path = scope.get("path", "")
        if curr_path.startswith("/api/index.py"):
            scope["path"] = curr_path[len("/api/index.py"):] or "/"
            
    await app(scope, receive, send)
