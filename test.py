name: Test Iris Model

on:
  workflow_run:
    workflows: ["Train Iris Model"]  # Ensure the workflow name matches
    branches:
      - main
    types:
      - completed

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      # Checkout the repository
      - name: Checkout code
        uses: actions/checkout@v3

      # Download the trained model artifact from the previous pipeline
      - name: Download model artifact
        uses: actions/download-artifact@v3
        with:
          name: iris-model
          path: ./  # Downloads the artifact to the current directory

      # Add the testing data file
      - name: Add test data
        run: |
          echo '{ "data": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 5.1, 1.8]], "expected": [0, 2, 2] }' > data.json

      # Setup Python 3.10
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      # Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # Run test script
      - name: Run tests
        run: python test.py
