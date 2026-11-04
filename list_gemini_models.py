"""List Gemini models your key can call.  Usage: GEMINI_API_KEY=... python list_gemini_models.py"""
import os, sys
import httpx

key = os.getenv("GEMINI_API_KEY")
if not key:
    sys.exit("Set GEMINI_API_KEY first")
r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models",
              headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=20)
print("HTTP", r.status_code)
if r.status_code != 200:
    sys.exit(r.text[:500])
for m in r.json().get("models", []):
    if "generateContent" in m.get("supportedGenerationMethods", []):
        print(m["name"].replace("models/", ""))
