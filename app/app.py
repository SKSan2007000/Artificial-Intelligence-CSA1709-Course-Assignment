from flask import Flask,render_template,request,jsonify
import os,io,time,statistics
try:
    from .engine import *
    from .data import CANDIDATES, JOBS
except ImportError:
    from engine import *
    from data import CANDIDATES, JOBS
app=Flask(__name__, template_folder='../templates')

def profiles(): return [{**c,**parse_resume(c['text'])} for c in CANDIDATES]
def jobs(): return [{**j,**parse_job(j['text'])} for j in JOBS]
def train():
 rows=[]; cs=profiles(); js=jobs()
 for c in cs:
  for j in js:
   w,_=weighted_match(c,j); rows.append({'candidate':c,'job':j,'label':1 if w>=65 else 0})
 return rows
ROWS=train(); CLF=decision_tree_train(ROWS)

@app.route('/')
def index(): return render_template('index.html')
@app.route('/api/data')
def data():
 cs=profiles(); js=jobs(); return jsonify({'candidates':cs,'jobs':js,'stats':{'candidates':len(cs),'jobs':len(js),'skills':len(SKILL_ALIASES),'algorithms':5}})
@app.route('/api/dashboard')
def dashboard():
 cs=profiles(); js=jobs(); scores=[]
 for c in cs:
  for j in js:scores.append(screen_candidate(c,j,CLF)['final'])
 return jsonify({'avg_score':round(statistics.mean(scores),2),'top_score':max(scores),'candidate_count':len(cs),'job_count':len(js),'protected_firewall':'ACTIVE','human_review':'REQUIRED'})
@app.route('/api/screen',methods=['POST'])
def screen():
 d=request.get_json(); c=next((x for x in profiles() if x['id']==d.get('candidate_id')),None); j=next((x for x in jobs() if x['id']==d.get('job_id')),None)
 if not c or not j:return jsonify({'error':'Invalid selection'}),400
 r=screen_candidate(c,j,CLF); r['skill_gap_plan']=skill_gap_plan(c,j); return jsonify(r)
@app.route('/api/recommend',methods=['POST'])
def recommend():
 d=request.get_json(); c=next((x for x in profiles() if x['id']==d.get('candidate_id')),None)
 if not c:return jsonify({'error':'Invalid candidate'}),400
 return jsonify(recommend_jobs(c,jobs(),CLF))
@app.route('/api/compare',methods=['POST'])
def compare():
 d=request.get_json(); c=next((x for x in profiles() if x['id']==d.get('candidate_id')),None); j=next((x for x in jobs() if x['id']==d.get('job_id')),None)
 if not c or not j:return jsonify({'error':'Invalid selection'}),400
 r=screen_candidate(c,j,CLF); return jsonify({'weighted':r['weighted'],'tfidf':r['tfidf'],'greedy':r['greedy'],'astar':r['astar'],'hybrid':r['final'],'trace':r['astar_trace']})
@app.route('/api/logic',methods=['POST'])
def logic():
 d=request.get_json(); c=next((x for x in profiles() if x['id']==d.get('candidate_id')),None); j=next((x for x in jobs() if x['id']==d.get('job_id')),None)
 if not c or not j:return jsonify({'error':'Invalid selection'}),400
 cs=set(c['skills']); js=set(j['skills']); rules=[]
 for s in sorted(cs&js):rules.append(f'hasSkill(Candidate,{s}) ∧ requiresSkill(Job,{s}) → SkillMatch({s})')
 if c['years']>=j['years']:rules.append('experience(Candidate) ≥ requiredExperience(Job) → ExperienceSatisfied')
 if len(cs&js)>=3:rules.append('skillOverlap ≥ 3 → StrongCandidateEvidence')
 conclusion='Recommend for recruiter review' if len(rules)>=2 else 'Needs additional review'
 return jsonify({'facts':[f'hasSkill(Candidate,{s})' for s in sorted(cs)],'requirements':[f'requiresSkill(Job,{s})' for s in sorted(js)],'rules':rules,'conclusion':conclusion})
@app.route('/api/interview',methods=['POST'])
def interview():
 d=request.get_json(); qs=interview_questions(d.get('role','Software Engineer'),d.get('skills',[])); return jsonify({'questions':qs,'rubric':['technical correctness','reasoning','trade-off awareness','communication','evidence']})
@app.route('/api/skillgap',methods=['POST'])
def skillgap():
 d=request.get_json(); c=next((x for x in profiles() if x['id']==d.get('candidate_id')),None); j=next((x for x in jobs() if x['id']==d.get('job_id')),None)
 if not c or not j:return jsonify({'error':'Invalid selection'}),400
 return jsonify(skill_gap_plan(c,j))
@app.route('/api/upload',methods=['POST'])
def upload():
 f=request.files.get('resume');
 if not f:return jsonify({'error':'Upload a PDF, DOCX or TXT resume'}),400
 raw=f.read(); name=f.filename.lower(); text=''
 try:
  if name.endswith('.txt'):text=raw.decode('utf8','ignore')
  elif name.endswith('.docx'):
   from docx import Document; text='\n'.join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
  elif name.endswith('.pdf'):
   from pypdf import PdfReader; text='\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(raw)).pages)
  else:return jsonify({'error':'Only PDF, DOCX and TXT are supported'}),400
 except Exception as e:return jsonify({'error':str(e)}),400
 p=parse_resume(text); p['protected_firewall']=protected_flags(text); return jsonify(p)
@app.route('/api/benchmark')
def benchmark():
 cs=profiles(); js=jobs(); rows=[]
 for c in cs:
  for j in js:
   r=screen_candidate(c,j,CLF); rows.append(r)
 return jsonify({'count':len(rows),'averages':{k:round(statistics.mean([r[k] for r in rows]),2) for k in ['weighted','tfidf','greedy','astar','final']},'best_hybrid':max(r['final'] for r in rows)})
if __name__=='__main__':app.run(host='127.0.0.1',port=int(os.getenv('PORT','5055')),debug=False)
