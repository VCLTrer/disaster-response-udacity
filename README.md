
# Disaster Response Pipeline Project

This project aims to build a machine learning pipeline to categorize real messages sent during disaster events. The project includes an ETL pipeline to process data, an ML pipeline to train a classifier, and a web app to visualize and interact with the model.

## Project Structure

```
DISASTER-RESPONSE-UDACITY
├── .github
│   └── workflows
│       └── pylint.yml
├── app
│   ├── templates
│   │   ├── go.html
│   │   └── master.html
│   └── run.py
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   └── process_data.py
├── models
│   ├── classifier.pkl
│   └── train_classifier.py
├── .gitattributes
└── README.md
```

## Components

### 1. ETL Pipeline
The ETL pipeline processes the raw data and stores it in a SQLite database.
- **File**: `data/process_data.py`
- **Functionality**:
  - Loads messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores the cleaned data in a SQLite database

### 2. ML Pipeline
The ML pipeline trains a machine learning model to classify disaster messages.
- **File**: `models/train_classifier.py`
- **Functionality**:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file

### 3. Flask Web App
The web app visualizes the data and allows users to input new messages for classification.
- **Directory**: `app`
- **Files**:
  - `run.py`: Flask application
  - `templates/`: HTML templates for the web app

## Getting Started

### Prerequisites
- Python 3.x
- Required Python libraries: pandas, numpy, sqlalchemy, scikit-learn, nltk, flask, plotly

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-repo/disaster-response-pipeline.git
   cd disaster-response-pipeline
   ```

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

### Running the Pipelines and Web App

1. **Run the ETL pipeline**:
   ```
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```

2. **Run the ML pipeline**:
   ```
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

3. **Run the web app**:
   ```
   cd app
   python run.py
   ```

4. **View the web app**:
   - Click the `PREVIEW` button in your IDE (if applicable) or open your web browser and go to `http://localhost:3001/`

## Project Motivation
The project is part of the Udacity Data Scientist Nanodegree Program. It aims to apply data engineering skills to process disaster response data and build a machine learning pipeline to categorize emergency messages. This helps disaster response organizations prioritize and allocate resources effectively during disaster events.

## Acknowledgments
- Udacity for providing the project template and datasets.
- Figure Eight for providing the disaster response message dataset.

## License
This project is licensed under the MIT License.

---

This README provides a comprehensive overview of the project, including its structure, components, prerequisites, installation instructions, and usage guidelines.