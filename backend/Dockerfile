# We're using the official Python image from the Docker Hub
FROM python:3.10

# Set a directory for our application
WORKDIR /usr/src/app

# Copy requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the container
COPY . .

# Specify the command to run when the container starts
CMD [ "gunicorn", "app:app", "-b", "0.0.0.0:5000"]