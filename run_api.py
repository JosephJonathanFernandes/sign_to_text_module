"""
Launcher for the ISL Sign-to-Text API.

Usage:
    python run_api.py

Environment Variables:
    DEBUG=true    Enable top-5 debug responses
    PORT=8000     Change listening port
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    print("====================================================================")
    print(f"  Starting ISL API on port {port}")
    if debug_mode:
        print("  [WARNING] DEBUG mode is ON (sending full probability tensors)")
    print("====================================================================")
    
    # Run uvicorn programmatically
    # using string reference 'api.app:app' allows uvicorn to reload if needed
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=False)
