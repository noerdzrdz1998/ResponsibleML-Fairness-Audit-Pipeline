# Lightweight Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . .

# Create the folders the pipeline expects
RUN mkdir -p predictions models audit

# Run the debiasing script, then the audit
CMD ["bash", "-c", "python Fair_machine_learning.py && python audit.py"]
