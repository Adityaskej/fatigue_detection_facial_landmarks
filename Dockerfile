# ---------------------------
# 1. Base Image
# ---------------------------
FROM python:3.10-slim

# ---------------------------
# 2. Working Directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 3. Copy Requirements
# ---------------------------
COPY requirements.txt .

# ---------------------------
# 4. Install Dependencies
# ---------------------------
RUN pip install --no-cache-dir -r requirements.txted

# ---------------------------
# 5. Copy the Entire App
# ---------------------------
COPY . .

# ---------------------------
# 6. Expose Streamlit Default Port
# ---------------------------
EXPOSE 8501

# ---------------------------
# 7. Run Streamlit App

# ---------------------------
CMD ["streamlit", "run", "fatigue_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
