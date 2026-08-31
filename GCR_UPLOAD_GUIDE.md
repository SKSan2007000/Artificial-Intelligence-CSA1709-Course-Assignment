# GitHub / Google Cloud Run upload guide

## GitHub
```bash
git init
git add .
git commit -m "TalentLens AI assignment implementation"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY>
git push -u origin main
```

## Local verification
```bash
pip install -r requirements.txt
python tests/test_engine.py
python evaluate.py
python app/app.py
```

## Cloud Run
Build and deploy the container using the Google Cloud project configured by your faculty/team. The Flask app currently listens on port 5055; for Cloud Run, set the container port to 5055 or adapt the app to read the PORT environment variable.

## Evidence to capture
1. GitHub repository URL and commit.
2. Test output showing `3/3 tests passed`.
3. Browser screenshots of Dashboard, Candidate Screening, Job Recommender, Knowledge Reasoning and Mock Interview.
4. Cloud Run URL and successful page load, if deployment is required.
