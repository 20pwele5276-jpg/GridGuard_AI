# GridGuard AI

GridGuard AI is an electrical equipment monitoring and maintenance support application built with Python.

It helps users explore electrical readings, identify unusual patterns, review measurement trends, search equipment manuals, and generate maintenance reports.

## Features

- CSV and Excel data upload
- Downloadable CSV template
- Sample analysis for users without their own data
- Electrical reading validation
- Anomaly detection using Isolation Forest
- Voltage and temperature trend charts
- Relative anomaly scores
- PDF maintenance reports
- CSV analysis downloads
- PDF equipment manual upload
- Semantic search using Sentence Transformers and FAISS
- LLM-based explanations using the Groq API
- Source filenames and page references for retrieved manual passages
- Three-page navigation: Dashboard, How to Use, and Try Sample Analysis
- Automated tests using pytest

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Sentence Transformers
- FAISS
- PyPDF
- Groq API
- ReportLab
- Pytest

## Project Structure

```text
GridGuard_AI/
├── app/
│   ├── ui.py
│   └── services/
│       ├── anomaly_detector.py
│       ├── manual_search.py
│       ├── llm_service.py
│       └── report_generator.py
├── data/
├── logs/
├── reports/
├── tests/
│   ├── test_basic.py
│   └── test_anomaly_detector.py
├── .gitignore
├── requirements.txt
└── README.md