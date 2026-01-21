# House Price Prediction - Machine Learning Regression Project

A machine learning project to predict median house values in California using regression techniques. This project compares **Linear Regression** and **Random Forest** models.

## Project Structure
* `data/`: Contains the dataset (`housing.csv`).
* `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and Model Training.
* `src/`: Custom Python package for data preprocessing pipelines and model training.

## Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YElshahry/House-Price-Prediction-ML-Regression-Project.git](https://github.com/YElshahry/House-Price-Prediction-ML-Regression-Project.git)
    cd House-Price-Prediction-ML-Regression-Project
    ```

2.  **Install dependencies and the project package:**
    *(This installs the `src` folder as a package so imports work correctly)*
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

##  Usage
* **Run the Modelling Notebook:**
    Open `notebooks/02_modelling.ipynb` to see the full training and evaluation process.

* **Train via Script:**
    You can also train the model directly from the terminal:
    ```bash
    python src/model_training.py
    ```

## Results
The **Random Forest Regressor** achieved the best performance with an RMSE of approximately **$50,000** on the test set, significantly outperforming the Linear Regression baseline.