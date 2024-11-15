#!/bin/bash

# Check if venv folder exists
if [ ! -d "venv" ]; then
  echo "Virtual environment not found. Please create it using 'python3 -m venv venv'."
  exit 1
fi

# Activate the virtual environment
source venv/bin/activate

if [ $? -ne 0 ]; then
  echo "Failed to activate virtual environment. Make sure you're using the correct shell and venv path."
  exit 1
fi

echo "Virtual environment activated."

# Run the Django development server
echo "Starting the Django development server..."
python manage.py runserver

# Deactivate the virtual environment
deactivate
