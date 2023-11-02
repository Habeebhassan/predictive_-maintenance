# Using the base image with Python 3.10
FROM python:3.10

# Set our working directory as app
WORKDIR /code
# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Installing Python packages through requirements.txt file
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application files (server.py, models, HTML, etc.) into the container
COPY . .

# Exposing port 5000 from the container
EXPOSE 7860

# Starting the Python application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

