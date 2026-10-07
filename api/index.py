import os
import sys

# Append parent directory to sys.path so app modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# For Vercel Python Serverless Function
if __name__ == "__main__":
    app.run()
