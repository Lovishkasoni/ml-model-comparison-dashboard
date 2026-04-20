FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ /app/backend/

# Copy frontend
COPY frontend/ /app/frontend/

# Expose ports
EXPOSE 5000

# Run Flask app
CMD ["python", "/app/backend/app.py"]