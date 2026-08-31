import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","app"))
from engine import *
from data import CANDIDATES,JOBS

def test_exact_match():
 c=parse_resume(CANDIDATES[0]["text"]); j=parse_job(JOBS[0]["text"]); assert screen_candidate(c,j)["final"] > 70
def test_missing_skill():
 c=parse_resume("Python developer with 3 years experience. B.Tech Computer Science. Python, SQL."); j=parse_job("Backend role requiring Python, SQL, Docker, AWS. Minimum 2 years."); r=screen_candidate(c,j); assert "docker" in r["missing"] and "aws" in r["missing"]
def test_astar():
 c=parse_resume(CANDIDATES[2]["text"]); j=parse_job(JOBS[2]["text"]); assert screen_candidate(c,j)["astar"] >= 80
def test_greedy_exposed():
 c=parse_resume(CANDIDATES[0]["text"]); j=parse_job(JOBS[0]["text"]); r=screen_candidate(c,j); assert 0 <= r["greedy"] <= 100
def test_astar_trace():
 c=parse_resume(CANDIDATES[0]["text"]); j=parse_job(JOBS[0]["text"]); r=screen_candidate(c,j); assert len(r["astar_trace"]) == len(j["skills"])
def test_protected_firewall():
 r=parse_resume("Female candidate, age 21, Python developer with 2 years experience"); assert "python" in r["skills"]
def test_recommendation_fields():
 c=parse_resume(CANDIDATES[0]["text"]); out=recommend_jobs(c,[{**j,**parse_job(j["text"])} for j in JOBS]); assert out and "matched" in out[0] and "missing" in out[0]
def test_skill_gap():
 c=parse_resume("Python developer"); j=parse_job("Backend Developer requiring Python, Docker, AWS"); assert any(x["skill"] in ["docker","aws"] for x in skill_gap_plan(c,j))
if __name__=="__main__":
 tests=[test_exact_match,test_missing_skill,test_astar,test_greedy_exposed,test_astar_trace,test_protected_firewall,test_recommendation_fields,test_skill_gap]
 for t in tests:t()
 print(f"{len(tests)}/{len(tests)} tests passed")
