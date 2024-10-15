# Phase 1: Build stage
FROM python:3.9-alpine AS build

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache build-base libffi-dev

# Copy the requirements.txt file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Phase 2: Final stage
FROM python:3.9-alpine

# Set the working directory
WORKDIR /app

# Copy dependencies from the build stage
COPY --from=build /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "your_script.py"]

# Steps to Build and Run the Docker Container:

#     Build the Docker Image:
    
#     docker build -t qa-bot .

#     Run the Docker Container:

#     docker run -p 8501:8501 qa-bot

#     Visit http://localhost:8501 in your browser to see the running app.