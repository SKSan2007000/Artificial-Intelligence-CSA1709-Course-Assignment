# TalentLens AI v3.0

Explainable AI resume screening and job recommendation system for CSA1709 Artificial Intelligence.

## Core AI components
- Structured resume/job extraction
- Weighted requirement matching
- TF-IDF cosine similarity
- Greedy Best-First Search comparison
- A* informed requirement search with visible g/h/f trace
- Decision Tree suitability model
- Propositional/FOL-style recruitment reasoning
- Explainable score decomposition and skill gaps

## Product modules
- Command Center
- Candidate Screening
- Job Radar
- A* Laboratory
- Logic Engine
- Skill Gap Planner
- Mock Interview Studio
- Resume Ingestion / ATS audit

## Run
```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest tests\test_engine.py -v
python app\app.py
```
Open http://127.0.0.1:5055

## Cloud Run
A Dockerfile is included. The Flask app reads PORT from the environment.

## Academic note
The bundled benchmark is synthetic and for reproducible educational validation. It must not be interpreted as evidence of production hiring accuracy. Human review remains mandatory.
