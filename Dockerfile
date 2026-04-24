# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Ensures logs appear immediately (important for debugging)
ENV PYTHONUNBUFFERED=1

# Copy dependency list and install them
# Copying requirements first leverages Docker's layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and the pre-trained model
# COPY artifacts/ ./artifacts
COPY . .

# Expose the port the app runs on (e.g., 8000 for FastAPI, 5000 for Flask)
EXPOSE 8000

# Start the application using a production-grade server like Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
