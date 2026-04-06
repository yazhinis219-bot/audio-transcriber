import uvicorn
import os

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.gateway.main:app", host="0.0.0.0", port=port, reload=os.getenv("RELOAD", "0") == "1")
