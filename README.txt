WATER QUALITY AND POTABILITY PREDICTION
Using Supervised Machine Learning (Binary Classification)

PROJECT ID: AI_20
AI_20_Report.pdf 
Team members: N. SAI MANVITHA , S. VIPLAV KUMAR.
Enrollment numbers: 24STUCHH010978 , 24STUCHH010922.
Tested with: Python 3.15.2 on macOS


1. PROBLEM STATEMENT
--------------------
Drinking water must be checked for chemical and physical properties before
it is considered suitable for human consumption. Laboratory testing is
accurate but slow and costly for routine monitoring. This project explores
whether a supervised machine learning model can learn patterns from recorded
water samples and predict whether a sample is Potable (1) or Not Potable (0)
from eight water quality measurements.


2. PROJECT DESCRIPTION
----------------------
- Type: Supervised learning, binary classification.
- Data: A labelled dataset of 3,276 water samples.
- Preprocessing: Duplicate check, treatment of pH = 0 as missing, median
  imputation and standard scaling, all fitted on training data only.
- Split: Stratified 80:20 train-test split (2,620 training rows, 656 testing
  rows), random seed 42.
- Models: Logistic Regression, Decision Tree and Random Forest, each tuned
  with 5-fold stratified cross-validation on the training set using F1 for
  the Potable class. Class weights are set to "balanced" to handle the
  imbalance (61% Not Potable, 39% Potable).
- Selection: Random Forest was selected for the highest cross-validated F1
  on the training data.
- Interface: A Streamlit web application where the user enters eight values
  and receives the predicted class, the probability of Potable, and the model
  name.

Important: This is an educational machine learning project. The predictions
are learned from a dataset and do NOT certify that real water is safe to
drink. Use accredited laboratory testing for any drinking-water decision.


3. DATASET
----------
Name      : Water Potability dataset
Source    : Kaggle, by Aditya Kadiwal
URL       : https://www.kaggle.com/datasets/adityakadiwal/water-potability
File      : water_potability.csv (included in this folder)
Records   : 3,276 rows, 10 columns
Missing   : ph 491 (15.0%), Sulfate 781 (23.8%), Trihalomethanes 162 (4.9%)
Duplicates: 0
Classes   : 0 Not Potable = 1,998 (61.0%); 1 Potable = 1,278 (39.0%)

Column descriptions:
  ph               : pH of the water (0 to 14). Values of 0 are treated as
                     missing because they are physically implausible.
  Hardness         : Capacity of water to precipitate soap, mainly due to
                     calcium and magnesium (mg/L).
  Solids           : Total dissolved solids (ppm).
  Chloramines      : Amount of chloramines disinfectant (ppm).
  Sulfate          : Amount of dissolved sulfate (mg/L).
  Conductivity     : Electrical conductivity of the water (microS/cm).
  Organic_carbon   : Total organic carbon (ppm).
  Trihalomethanes  : Amount of trihalomethanes (microg/L).
  Turbidity        : Light-scattering measure of cloudiness (NTU). Kept in
                     the data and EDA plots, but NOT used for modelling,
                     following the eight input parameters in the project brief.
  Potability       : Target label. 1 = Potable, 0 = Not Potable.

The eight model inputs are: ph, Hardness, Solids, Chloramines, Sulfate,
Conductivity, Organic_carbon, Trihalomethanes.


4. SOFTWARE REQUIREMENTS
------------------------
- macOS or Windows laptop (tested on macOS)
- Python 3.15.2 (tested). Python 3.11 or 3.12 should also work.
- Visual Studio Code
- Internet connection for the one-time library installation
- Libraries listed in requirements.txt:
    pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, streamlit


5. INSTALLATION
---------------
Open the project folder in VS Code (File > Open Folder), then open the
Terminal (Terminal > New Terminal) and run these commands one at a time:

  python3 -m venv venv
  source venv/bin/activate           (Windows: venv\Scripts\activate)
  pip install -r requirements.txt

The prompt should show (venv) after activation.


6. HOW TO RUN
-------------
Step A: Generate the EDA charts (optional, saves 5 files in figures/):
  python eda_plots.py

Step B: Train, tune, evaluate and save the models (about 10 seconds):
  python water_quality_prediction.py

  This creates:
    best_model.joblib            (saved selected model)
    model_results.csv            (all metrics, train and test)
    figures/confusion_matrix.png
    figures/model_comparison.png

Step C: Start the website:
  streamlit run app.py

  Open http://localhost:8501 if the browser does not open automatically.
  Enter the eight values and click Predict. To stop the website, press
  Control + C in the terminal.

Optional: python preprocessing.py prints the cleaning report and the
train-test split without training any model.


7. EXPECTED OUTPUT
------------------
Test set results (20% hold-out, 656 samples):

  Model                       Accuracy  Bal.Acc  F1 (Potable)  ROC-AUC
  Logistic Regression          0.529     0.529     0.468        0.548
  Decision Tree                0.587     0.584     0.519        0.636
  Random Forest (selected)     0.637     0.609     0.508        0.673
  Baseline: always Not Potable 0.610     0.500     0.000        0.500
  Baseline: always Potable     0.390     0.500     0.561        0.500

Selected model: Random Forest (highest 5-fold CV F1 on training data = 0.5463)

Interpretation: The water quality features carry only a weak signal for
potability. Random Forest ranks samples better than chance and beats the
Not Potable baseline on accuracy and balanced accuracy, but it does not beat
the always-Potable baseline on F1. Training scores are much higher than test
scores, which shows overfitting. These limitations are discussed in the
project report.


8. SCREENSHOTS
--------------
[Screenshot placeholder: screenshots/02_website_prediction.png]
Website showing a prediction for the sample values.

[Screenshot placeholder: screenshots/03_website_validation.png]
Website showing the validation error for an invalid entry.

[Screenshot placeholder: screenshots/04_training_results.png]
Terminal output showing the training and test results table.

[Charts: figures/model_comparison.png and figures/confusion_matrix.png]


9. PROJECT STRUCTURE
--------------------
PROJECT_ID/
├── README.txt                    Project documentation (this file)
├── requirements.txt              Pinned Python libraries
├── water_potability.csv          Dataset
├── preprocessing.py              Cleaning, split and preprocessing pipeline
├── eda_plots.py                  Exploratory data analysis charts
├── water_quality_prediction.py   Model training, evaluation and saving
├── app.py                        Streamlit web interface
├── best_model.joblib             Saved trained model (Random Forest)
├── model_results.csv             Train and test metrics
├── figures/                      Generated charts
└── screenshots/                  Website and terminal screenshots


10. HOW TO LOAD THE SAVED MODEL IN PYTHON
-----------------------------------------
  import joblib
  bundle = joblib.load("best_model.joblib")
  pipeline = bundle["pipeline"]     # includes imputation and scaling
  # bundle["features"] lists the eight input columns in order
  # pipeline.predict_proba(X)[:, 1] gives the probability of Potable