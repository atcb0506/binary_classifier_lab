# base image
FROM python:3.7-stretch

# Set up kernel
RUN apt-get update

# Add codes to container
ADD py_app /app

# Install Python requirements
RUN pip install -r /app/criteo_ads_data/requirements.txt

# Packaging app
WORKDIR app
