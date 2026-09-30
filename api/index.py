import os
import sys

# Ensure backend modules can be imported when running as a Vercel Serverless Function
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.main import app

# Export app for Vercel WSGI/ASGI handler
handler = app
