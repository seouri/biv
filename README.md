# Height Error Labeling Dashboard

Interactive application for reviewing and labeling errors in pediatric longitudinal height measurements.

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Authentication](#authentication)
- [Project Structure](#project-structure)
- [Input Data](#input-data)
- [Output Data](#output-data)

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+**
- **uv** package manager

### Setup Steps

1. **Clone or download this repository**

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

3. **Verify installation**:
   ```bash
   uv run streamlit --version
   ```

---

## 💻 Usage

### Starting the Application

Run the dashboard using uv:
```bash
uv run streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`.

### 🔐 Authentication

The application requires login for user-specific data management. Each user's labels and processed data are stored separately.

**Available Users:**
- `user1`
- `user2`
- `user3`
- `user4`
- `user5`

**User-Specific Data Storage:**
- Labels are saved to: `data/labels/{username}/`
- Processed data exports to: `data/processed/{username}/`


### Using the Dashboard

1. **Select a Patient**: Use the sidebar to choose a patient from the dropdown menu, or click individual patient button
2. **Review Measurements**: Click on data points in the growth charts or use ◀/▶ arrows to navigate, or click on a row in the data table
3. **Mark Errors**: Click "Mark as Error" for problematic measurements
4. **Add Comments**: Provide specific comments for individual points or general notes for the entire patient
5. **Complete Review**: Click "Mark Patient as Complete" when finished reviewing all measurements for a patient
6. **Export Data**: Use the "💾 Save Labeled Data" button in the sidebar to save and optionally download results

---

## 📁 Project Structure

```
biv/
│
├── main.py                      # Application entry point
├── pyproject.toml               # Project dependencies and configuration
├── README.md                    # This file
│
├── data/                        # Data directory
│   ├── raw/                     # INPUT: Place your patient data here
│   │   ├── visits_60_patients.csv        # Patient visit data sample (first 60 patients)
│   │
│   ├── growth_standard/         # Reference growth standards (DO NOT MODIFY)
│   │   ├── who_growth_standards.csv      # WHO growth charts
│   │   ├── statage_combined.csv          # CDC height-for-age standards
│   │   ├── bmiagerev.csv                 # CDC BMI standards
│   │   └── ...                           # Other reference files
│   │
│   ├── labels/                  # INTERMEDIATE OUTPUT: Individual patient label files (JSON) -- for data persistence
│   │   ├── Pxxxxxx_labels.json
│   │   └── Pxxxxxy_labels.json
│   │
│   └── processed/               # OUTPUT: Combined labeled datasets
│       └── all_patients_labeled.csv
│
└── src/                         # Source code
    ├── app.py                   # Main application logic
    ├── config.py                # Configuration constants
    │
    ├── components/              # UI components
    │   ├── sidebar.py           # Sidebar navigation
    │   ├── growth_chart.py      # Height-for-age chart
    │   ├── velocity_chart.py    # Growth velocity chart
    │   ├── data_table.py        # Measurement data table
    │   └── ...
    │
    ├── data/                    # Data handling
    │   ├── loader.py            # Data loading functions
    │   ├── processor.py         # Data preprocessing
    │   └── growth_standards.py  # Z-score calculations
    │
    ├── utils/                   # Utility functions
    │   ├── calculations.py      # Growth velocity & metrics
    │   ├── persistence.py       # Save/load labels
    │   └── state_manager.py     # Session state management
    │
    └── styles/
        └── custom.css           # Custom styling
```

---

## 📥 Input Data

### Required Input Format

Place your patient data CSV file(s) in the `data/raw/` directory.

**Required Columns:**
- `patient_id` - Unique patient identifier
- `visit_date` - Date of visit (any parseable date format)
- `age_in_days` - Patient age in days at visit
- `height_in` - Height measurement in inches
- `weight_oz` - Weight measurement in ounces (optional, for BMI calculations)
- `sex` - Patient sex ('M' or 'F')

**Example:**
```csv
patient_id,visit_date,age_in_days,height_in,weight_oz,sex
Pxxxxxx,xxxx-xx-xx,xx,xx,xx,xx
Pxxxxxx,xxxx-xx-xx,xx,xx,xx,xx
```

### Modifying Data Source

By default, the app loads `data/raw/visits_60_patients.csv`. To change this:

1. Edit `src/data/loader.py`, line ~65
2. Update the file path in the `load_patient_data()` function

---

## 📤 Output Data

The application generates two types of output files:

### 1. Individual Label Files (JSON)
**Location:** `data/labels/`

**Format:** `{patient_id}_labels.json`

**Contents:**
```json
{
  "patient_id": "Pxxxxxx",
  "error_indices": [5, 12],
  "point_comments": {
    "5": "Implausible growth spurt",
    "12": "Possible recording error"
  },
  "general_comment": "Overall growth pattern looks normal except for noted outliers",
  "completed": true,
  "timestamp": "2025-11-13T10:30:45"
}
```

### 2. Combined Labeled Dataset (CSV)
**Location:** `data/processed/all_patients_labeled.csv`

**Generated:** When you click "💾 Save Labeled Data" in the sidebar

**Contents:** 
- All original patient data columns
- `error` - Boolean flag indicating if the measurement was marked as an error
- `point_comment` - Specific comment for that measurement (if any)
- `general_comment` - General patient comment
- `completed` - Whether the patient review is complete

**Example:**
```csv
patient_id,visit_date,age_in_days,height_in,weight_oz,sex,error,point_comment,general_comment,completed
Pxxxxxx,xxxx-xx-xx,xx,xx,xx,xx,False,,,True
Pxxxxxx,xxxx-xx-xx,xx,xx,xx,xx,True,Implausible growth spurt,Overall normal,True
```

---