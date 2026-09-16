# Fraud Detection System

An end-to-end machine learning application that analyzes credit card transaction data and identifies transactions that exhibit patterns associated with fraud.

The project goes beyond training a machine learning model in isolation. It includes the complete path from data preprocessing and model training to a REST API, SQL prediction logging, an interactive React dashboard, and containerized deployment using Docker.

> **Current status:** The application is fully functional locally using Docker and Docker Compose. AWS deployment is planned but is not currently part of the deployed application.

## Project Summary

Credit card fraud detection is essentially a "needle in a haystack" problem.

A financial institution may process hundreds of thousands or millions of legitimate transactions while fraudulent transactions represent only a tiny fraction of the total. Manually reviewing every transaction is impractical, so automated systems can help identify transactions that appear suspicious enough to warrant further attention.

This project explores that problem using machine learning.

The model was developed using the **Credit Card Fraud Detection dataset** made available by the Machine Learning Group at the Université Libre de Bruxelles (ULB). The dataset contains **284,807 European credit card transactions**, of which only **492 are classified as fraudulent**.

That means fraud represents approximately **0.17% of the dataset**.

Because financial transaction information is sensitive, most of the original transaction attributes were transformed into anonymized numerical features using **Principal Component Analysis (PCA)**. The public dataset contains these as `V1` through `V28`, along with the original `Time`, `Amount`, and `Class` fields.

The extreme imbalance between legitimate and fraudulent transactions creates an important machine learning challenge. A model that classified every transaction as legitimate would achieve greater than 99% accuracy while detecting **zero fraud**.

For that reason, this project focuses on metrics such as **precision, recall, F1 score, and ROC-AUC**, rather than relying on accuracy alone.

# Machine Learning Terminology

First lets get some terminology out of the way in case you don't have a machine learning background. I was learning a lot throughout this process as well.

## Feature

A **feature** is a piece of information given to a machine learning model to help it make a prediction.

For this dataset, the model receives 30 features:

- `Time`
- `V1` through `V28`
- `Amount`

The `V1-V28` features are anonymized transformations of the original financial information.

---

## Label / Class

The **label** is the answer associated with a historical example.

For this dataset:

- `Class = 0` means the transaction was legitimate.
- `Class = 1` means the transaction was fraudulent.

## Time and Amount
- **Time** — The number of **seconds elapsed between a transaction and the first transaction recorded in the dataset**. It is a relative timestamp rather than a conventional date or clock time. For example, a value of `3600` means that the transaction occurred approximately one hour after the first recorded transaction.
-  **Amount** — The **monetary value of the transaction**. This allows transaction size to be considered as one of the inputs to the model.

## Precision and Recall
- **Precision** - One part of the F1 score. For every time the model flags a transaction as a fraud how many were in fact fraudulent?
- **Recall** - The other part of the F1 score. For every instance of fraud, how many were flagged?
  
These 2 combined make up the F1 score, what you want is a system with a particularly high score as to avoid too many false positives or misses.

##  Deployment

The next planned stage is cloud deployment.

AWS EC2 has been selected as the initial deployment target because it provides a straightforward environment for running the existing Dockerized application.

The production deployment is intentionally listed as **planned work** rather than claiming that the application is currently hosted on AWS.

---

# Running the Project Locally

## Prerequisites

The easiest way to run the complete application is with:

- Git
- Docker Desktop
- Docker Compose

Docker Desktop includes the Docker tooling needed for the containerized version of the application.

> **Note:** The dataset and trained model artifacts may not be included directly in the Git repository because of their size. See the dataset/model setup sections below if they are absent from your clone.

---

## 1. Clone the Repository

Clone the repository and enter its directory:

    git clone <YOUR-REPOSITORY-URL>
    cd FraudDetectionSystem

Replace `<YOUR-REPOSITORY-URL>` with the actual GitHub repository URL.

---

## 2. Obtain the Dataset

The project uses the **Credit Card Fraud Detection** dataset published through Kaggle by the Machine Learning Group at ULB.

Download:

    creditcard.csv

and place it at:

    backend/data/creditcard.csv

The CSV is intentionally excluded from Git because it is a large external dataset.

---

## 3. Model Artifacts

The application requires:

    backend/models/fraud_model.pt
    backend/models/scaler.pkl

These contain the trained PyTorch model and fitted preprocessing scaler.

If model artifacts are distributed separately, place them in the locations above.

Alternatively, the project can be retrained using the included training pipeline to generate new artifacts.

> Because training can introduce differences between model runs, newly trained models may not produce exactly the same scores or evaluation results as the reference model documented in this README.

---

## 4. Start Docker Desktop

Make sure Docker Desktop is running before continuing.

On Windows, Docker Desktop may require hardware virtualization to be enabled.

Verify Docker is available with:

    docker --version

and:

    docker compose version

---

## 5. Build and Start the Application

From the project root, run:

    docker compose up --build

Docker Compose will build and start the frontend and backend services.

Once startup completes, open:

    http://localhost:3000

The FastAPI backend is available at:

    http://localhost:8000

FastAPI's interactive API documentation is available at:

    http://localhost:8000/docs

---

## 6. Test the Application

The React interface includes example transactions that can be loaded into the form.

Select either a known legitimate or known fraudulent example and submit it for analysis.

The dashboard should display:

- Risk score
- Classification
- Classification threshold

The request will also be written to the prediction database and should appear in the prediction history and dashboard statistics.

---

## 7. Stop the Application

Press:

    Ctrl + C

in the terminal running Docker Compose.

The application can also be stopped with:

    docker compose down

The prediction database uses persistent Docker storage, so prediction history can survive container replacement.

---

# Running Without Docker

The application can also be run directly during development.

The backend requires a Python virtual environment with the dependencies listed in:

    backend/requirements.txt

From the backend directory, activate the environment and start FastAPI:

    python -m uvicorn api:app --reload

The backend will run at:

    http://127.0.0.1:8000

For the frontend:

    cd frontend
    npm install
    npm run dev

Vite will normally make the development frontend available at:

    http://localhost:5173

Docker Compose is recommended when the goal is simply to run the complete application because it provides a more reproducible environment.

