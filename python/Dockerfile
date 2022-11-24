FROM python:3.7.3-stretch

# Maintainer info
LABEL maintainer="xq5wrx@gmail.com"

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install -r requirements.txt