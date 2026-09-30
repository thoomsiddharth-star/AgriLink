import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.main import app

async def handler(scope, receive, send):
    if scope.get("type") == "http":
        headers = dict(scope.get("headers", []))
        # Vercel supplies the real matched or invoked path in headers
        matched_path = (
            headers.get(b"x-matched-path", b"").decode("latin-1")
            or headers.get(b"x-invoke-path", b"").decode("latin-1")
            or headers.get(b"x-vercel-matched-path", b"").decode("latin-1")
            or headers.get(b"x-original-url", b"").decode("latin-1")
        )
        if matched_path:
            # Strip any query parameters if present in matched_path
            path_only = matched_path.split("?")[0]
            if not path_only.endswith("index.py"):
                scope["path"] = path_only
        elif scope.get("path", "").startswith("/api/index.py"):
            scope["path"] = scope["path"][len("/api/index.py"):] or "/"
            
    await app(scope, receive, send)
