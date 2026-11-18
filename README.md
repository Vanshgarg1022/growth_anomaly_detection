#👶 Paediatric Growth Anomaly Early Warning System

An AI-assisted paediatric growth-monitoring prototype and decision-support tool designed to identify emerging growth anomalies in children aged 0 to 5 years (0–1825 days) using longitudinal EMR data.Unlike conventional growth monitoring systems that rely solely on cross-sectional cut-offs (e.g., $Z < -2.0$), this system derives trajectory features—such as velocity/slope, standard deviation, and consecutive directional shifts—across repeated clinic visits. By analyzing longitudinal feature matrices, classical machine learning models (Random Forest) identify deteriorating growth patterns earlier than static WHO thresholds while enforcing strict data-quality validation and clinical auditability.📌 Project Summary & Key Performance BenchmarksTarget Population: Children aged 0–5 years based on official WHO Child Growth Standards.Engineered Feature Space: 52 time-aware longitudinal trajectory features (13 features per indicator across WAZ, HAZ, BMIZ, and HCZ) calculated strictly "as of" each visit to prevent look-ahead data leakage.Classifier Performance: The Random Forest model achieves an ROC-AUC of 0.9019 and 95.93% accuracy on held-out, child-disjoint test sets (compared to the conventional rule baseline ROC-AUC of 0.6144).Early Warning Advantage: Demonstrates a median lead time of 3.0 visits (95% CI: [2.0, 4.0]) over conventional static WHO cut-offs at matched false-alarm rates ($0.1963$).Synthetic Evaluation Benchmark: Evaluated on 2,000 synthetic children comprising 16,505 clinic visits across 17 distinct trajectory archetypes (including gradual weight faltering, progressive stunting, rapid weight loss, and recovery catch-up).Automated Test Coverage: Includes 89 automated unit and integration tests (pytest) verifying WHO LMS math precision, feature leakage prevention, model serialization, database operations, and PDF report creation.


#🏗️ System Architecture & Data Flow

+-----------------------------------------------------------------------+
| 1. PATIENT / VISIT ENTRY & SELECTION (Streamlit UI)                   |
| - New Patient (Atomic Code Sequence: A-0001, A-0002...)               |
| - Existing Patient (Demographic Look-up & Read-only Display)          |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 2. DATA-QUALITY & DYNAMIC VALIDATION LAYER                            |
| - Single-visit physical range checks                                  |
| - Age scope check (0 - 5 years / <= 1856 days)                        |
| - Sequential visit checks (Height drop > 1.0cm, HC drop > 0.5cm)      |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 3. WHO LMS GROWTH ENGINE (`who_zscore.py`)                            |
| - LMS Spline Interpolation against official WHO reference tables       |
| - Length/Height conversion offset (+/- 0.7 cm at 731 days)           |
| - Standardized z-score calculation: WAZ, HAZ, BMIZ, HCZ              |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 4. LONGITUDINAL TRAJECTORY FEATURE PIPELINE (`growth/features.py`)    |
| - Time-aware "as-of" feature matrix (52 indicators)                   |
| - Delta z, Slopes/month, Min/Max/SD, Largest Decline, Consecutive    |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------+-----------------------------------+
| 5a. CONVENTIONAL BASELINE         | 5b. SUPERVISED ML PIPELINE        |
| - Rules R1-R6 (Static Cut-offs)   | - Random Forest Classifier (v1)   |
| - Rule T1 (Trajectory Velocity)   | - Isolation Forest Anomaly Engine |
+-----------------------------------+-----------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 6. EXPLAINABLE AI & DECISION SUPPORT (`growth/explain.py`)            |
| - Risk Badges: Normal | Watch | Potential Growth Anomaly              |
| - Insufficient History Shield (< 3 visits)                            |
| - Top-3 Feature Driver Explanations (SHAP / Contribution weights)     |
+-----------------------------------------------------------------------+
                                   |
                                   v
+----------------------------------+------------------------------------+
| 7. AUDIT LOGGING (MySQL)         | 8. CLINICIAN REPORT EXPORT         |
| - `analysis_results` Table       | - ReportLab Engine PDF Export      |
| - Score, Version, Explanation    | - Matplotlib 2x2 Trajectory Graphs |
+----------------------------------+------------------------------------+


#🛠️ Technology Stack

Programming Language: Python 3.11+
User Interface: Streamlit (Multi-tab layout)
Database & Persistence: MySQL Server, SQLAlchemy ORM, mysql-connector-python
Numerical & Data Processing: Pandas, NumPy, SciPy (Cubic Splines), OpenPyXL
Machine Learning Pipelines: scikit-learn (Random Forest, Logistic Regression, Isolation Forest, GroupShuffleSplit), JoblibData 
Visualization: Plotly (Interactive UI charts), Matplotlib (Static figures for PDF generation)
Document Export Engine: ReportLab
Automated Testing: pytest


#📂 Repository File 

Directorygrowth_app/
├── .streamlit/
│   └── secrets.toml             # MySQL database credentials
├── data/                        # Synthetic datasets and feature matrix
│   ├── synthetic_children.csv   # Child profiles and target labels
│   ├── synthetic_errors.csv     # Controlled data-entry noise file
│   ├── synthetic_visits.csv     # Raw visit records across archetypes
│   └── asof_features.csv        # 52-column longitudinal feature matrix
├── docs/                        # Project documentation & reference assets
├── growth/                      # Core python engine package
│   ├── __init__.py
│   ├── baseline.py              # Rule-based conventional baseline (R1-R6, T1)
│   ├── early_warning.py         # Lead-time computation and bootstrap CIs
│   ├── explain.py               # Feature importance & alert explanation driver
│   ├── features.py              # Time-aware trajectory feature calculation
│   ├── inference.py             # App backend & MySQL audit log controller
│   ├── models.py                # Child-disjoint split & training routines
│   ├── report.py                # ReportLab PDF report generation engine
│   └── synth.py                 # Synthetic trajectory generation engine
├── models/
│   └── rf_v1.joblib             # Saved Random Forest model artifact
├── scripts/                     # Execution and management scripts
│   ├── load_synthetic_to_db.py  # Population script for MySQL database
│   ├── make_synthetic.py        # Dataset generation execution script
│   ├── run_phase2.py            # Feature engineering run script
│   ├── run_phase3.py            # Model training & evaluation run script
│   ├── run_phase4.py            # Early warning lead-time experiment script
│   └── seed_demo.py             # Demo profile seeding script (D-0001 to D-0005)
├── tests/                       # Unit & integration test suite (89 tests)
│   ├── test_app.py              # UI flow & input validation tests
│   ├── test_features_leakage.py # Feature leakage verification test
│   ├── test_models.py           # Model artifact & disjoint split tests
│   ├── test_phase4.py           # Early warning & CI bootstrap tests
│   ├── test_report.py           # PDF report engine generation tests
│   ├── test_synth.py            # Synthetic generator integrity tests
│   └── test_who_zscore.py       # WHO LMS calculation validation tests
├── who_tables/                  # Official WHO LMS reference Excel sheets (0-5 yrs)
├── app.py                       # Main 4-tab Streamlit web interface
├── db.py                        # Database connection setup
├── requirements.txt             # Python dependency manifest
├── trajectory.py                # Trajectory utility functions
└── who_zscore.py                # WHO z-score calculation module
🗄️ Database Schema DesignThe system relies on four primary MySQL tables within the growth_db database:1. patientsStores child demographic profiles.CREATE TABLE patients (
    patient_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_code VARCHAR(30) UNIQUE NOT NULL,
    date_of_birth DATE NOT NULL,
    sex VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
2. patient_code_sequenceManages atomic sequence generation for new patient registration (A-0001, A-0002, etc.).CREATE TABLE patient_code_sequence (
    sequence_id INT PRIMARY KEY DEFAULT 1,
    last_number INT NOT NULL DEFAULT 0
);
3. visitsStores longitudinal physical measurements and calculated $Z$-scores.CREATE TABLE visits (
    visit_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    visit_date DATE NOT NULL,
    weight_kg DECIMAL(5,2) NOT NULL,
    height_cm DECIMAL(5,2) NOT NULL,
    head_circumference_cm DECIMAL(5,2) NOT NULL,
    waz DECIMAL(5,2) NOT NULL,
    haz DECIMAL(5,2) NOT NULL,
    bmiz DECIMAL(5,2) NOT NULL,
    hcz DECIMAL(5,2) NOT NULL,
    measurement_type VARCHAR(20) NOT NULL,
    adjusted_length_height_cm DECIMAL(5,2) NOT NULL,
    bmi DECIMAL(5,2) NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
4. analysis_resultsPreserves an auditable trail of AI model runs, concern probabilities, and explanations.CREATE TABLE analysis_results (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    visit_id INT NOT NULL,
    method VARCHAR(30) NOT NULL,
    model_version VARCHAR(40) NOT NULL,
    risk_status VARCHAR(40) NOT NULL,
    score DECIMAL(6,4) NULL,
    n_valid_visits SMALLINT NOT NULL,
    explanation JSON NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id),
    UNIQUE KEY uq_visit_method_version (visit_id, method, model_version)
);
⚡ Installation & Setup Guide1. PrerequisitesPython 3.11+ installed on your system.MySQL Server (Community Edition or Workbench) running locally or remotely.2. Environment SetupClone the repository and create a Python virtual environment:git clone https://github.com/your-username/growth_app.git
cd growth_app

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
3. Database Setup & ConfigurationOpen MySQL Workbench and create a new schema named growth_db:CREATE DATABASE growth_db;
Configure database connection parameters inside .streamlit/secrets.toml:[mysql]
host = "localhost"
port = 3306
user = "your_mysql_username"
password = "your_mysql_password"
database = "growth_db"
4. Initialize & Train System PipelinesExecute the project initialization scripts in order:# 1. Generate synthetic dataset (2,000 children, 16,505 visits)
python scripts/make_synthetic.py

# 2. Extract longitudinal feature matrix
python scripts/run_phase2.py

# 3. Train Machine Learning models (saves models/rf_v1.joblib)
python scripts/run_phase3.py

# 4. Run early-warning lead-time evaluation
python scripts/run_phase4.py

# 5. Populate MySQL database with initial synthetic records
python scripts/load_synthetic_to_db.py

# 6. Seed demonstration patient profiles (D-0001 to D-0005)
python scripts/seed_demo.py
5. Run the ApplicationLaunch the Streamlit web dashboard:streamlit run app.py
🧪 Running Automated Unit TestsThe test suite covers numerical calculations, UI state handling, data leakage checks, model persistence, and PDF report creation:python -m pytest tests/
All 89 tests must pass successfully.🖥️ Live Demonstration Workflow (13 Steps)The application includes pre-seeded profiles (D-0001 to D-0005) for live demonstration:New Patient Registration: Switch to New Patient in Tab 1 (Entry); observe the sequence preview, fill measurements, and submit to auto-assign the next sequential code (A-####).Existing Patient Selection: Switch to Existing Patient in Tab 1; enter D-0001 to load pre-existing demographics (Gender, Date of Birth) automatically from MySQL.Data Quality Rejection: Attempt to submit a new visit with a height drop $> 1.0\text{ cm}$ below the previous visit. Observe the validation error blocking the write.WHO $Z$-Score Calculation: Inspect the automatically derived $Z$-scores (waz, haz, bmiz, hcz) on Tab 2 (History & Trajectory).Interactive Trajectory Plots: View 2x2 Plotly charts displaying historical $Z$-score progressions against standard WHO reference lines ($\pm 2\text{ SD}, \pm 3\text{ SD}$).Feature Matrix Extraction: Inspect engineered features (slopes, standard deviation, deltas) calculated under the hood.Conventional Rule Triggering: Navigate to Tab 3 (Risk Summary) for D-0004 to observe fired conventional rule IDs (e.g., R1, R6).AI Model Risk Scoring: View the Random Forest concern probability score and top-3 feature driver explanations on Tab 3.Early Warning Demonstration: Load child D-0001 (gradual weight faltering); observe that the ML model raises an anomaly alert 4 visits earlier than static cut-offs.Recovery Case Handling: Load child D-0002 (recovery catch-up); confirm the system categorizes the improving child as Normal/Watch rather than misclassifying them as deteriorating.Biological Implausibility Defense: Enter implausible values (e.g., 45 kg for a 1-year-old) to confirm the dynamic WHO flag limits block bad data.Insufficient History Shield: Load child D-0005 (1 visit); observe that the system displays Insufficient longitudinal history and suppresses trajectory predictions.Clinician PDF Summary Export: Go to Tab 4 (Report Export) and click Download PDF Report to export a formatted A4 clinician summary with embedded trajectory charts and safety disclaimers.⚠️ Scope, Safety Boundary & Clinical DisclaimerPrototype Scope: This application is a decision-support prototype validated against WHO Child Growth Standards for children aged 0 to 5 years. It is not designed or validated for older children (5–19 years).Non-Diagnostic Tool: The system does not diagnose clinical conditions (such as failure to thrive, growth hormone deficiency, or endocrine disorders). It provides risk stratification alerts to assist clinical evaluation.Clinician Authority: Model outputs and alert categories must be reviewed by a qualified healthcare professional. All predictions must be evaluated alongside clinical examination and context.Data Privacy: Synthetic patient data (S- and D- codes) were used for design and experimental validation. Real patient EMR data requires ethics approval and governance compliance prior to system integration.
