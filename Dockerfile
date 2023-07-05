# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Task_Queue directory to the container
COPY Task_Queue /app/Task_Queue

# Copy the PrivateData directory to the container
COPY PrivateData /app/PrivateData

# Copy the requirements.txt file
COPY Task_Queue/requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Run Celery worker
CMD celery -A Task_Queue.task_queue worker --loglevel=info


###docker build -t celery-image .

###docker run --name celery-container --link strange_bardeen:redis -d -v /c:/Users/natha/PycharmProjects/StockAlgoV2:/StockAlgoV2_mount -e TZ=America/New_York celery-image

