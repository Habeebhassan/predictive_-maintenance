# Using the base image with Python 3.10
FROM python:3.10

# Set our working directory as app
WORKDIR /app 

# Copy the requirements file into the container
COPY requirements.txt .

# Installing Python packages through requirements.txt file
RUN pip install -r requirements.txt || true

# Copy the rest of your application files (server.py, models, HTML, etc.) into the container
COPY . .

# Exposing port 5000 from the container
EXPOSE 5000

# Starting the Python application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]

