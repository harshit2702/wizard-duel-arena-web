FROM python:3.9

WORKDIR /code

# Copy requirements first for caching
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . /code

# Create a writable directory for the application to use if needed (e.g. for logs or temporary files)
RUN mkdir -p /code/logs && chmod 777 /code/logs

# Command to run the application
# connect strictly to port 7860 for Hugging Face Spaces
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860"]
