import re, math, heapq
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity

SKILL_ALIASES={
 'python':['python'],'java':['java'],'javascript':['javascript','js'],'typescript':['typescript','ts'],
 'react':['react','react.js'],'node':['node','node.js','nodejs'],'sql':['sql','mysql','postgresql','postgres'],
 'mongodb':['mongodb','mongo'],'docker':['docker','containerization'],'kubernetes':['kubernetes','k8s'],
 'aws':['aws','amazon web services'],'azure':['azure'],'gcp':['gcp','google cloud'],'git':['git','github','gitlab'],
 'fastapi':['fastapi'],'flask':['flask'],'django':['django'],'machine learning':['machine learning','ml'],
 'deep learning':['deep learning','neural network','pytorch','tensorflow'],'nlp':['nlp','natural language processing'],
 'computer vision':['computer vision','opencv','yolo'],'data analysis':['data analysis','pandas','numpy','power bi','tableau'],
 'cybersecurity':['cybersecurity','cyber security','information security'],'linux':['linux','ubuntu'],
 'rest api':['rest api','restful','api development'],'communication':['communication','presentation','public speaking'],
 'problem solving':['problem solving','analytical'],'gitops':['gitops'],'ci/cd':['ci/cd','continuous integration','continuous delivery'],
 'redis':['redis'],'kafka':['kafka'],'pytorch':['pytorch'],'tensorflow':['tensorflow'],'azure devops':['azure devops']
}
EDU_TERMS=['b.e','b.tech','be ','btech','b.sc','m.e','m.tech','computer science','information technology','engineering']
ROLE_TERMS=['software engineer','data analyst','data scientist','ml engineer','backend developer','frontend developer','full stack developer','devops engineer','cybersecurity analyst','ai engineer','cloud engineer','data engineer']
PROTECTED_TERMS=['gender','male','female','age','date of birth','religion','caste','marital status','nationality','disability','photo','address']

def clean(t): return re.sub(r'\s+',' ',(t or '').lower()).strip()

def extract_skills(text):
 t=clean(text); out=[]
 for s,aliases in SKILL_ALIASES.items():
  if any(re.search(r'(?<!\w)'+re.escape(a.lower())+r'(?!\w)',t) for a in aliases): out.append(s)
 return sorted(set(out))

def extract_years(text):
 vals=[]
 for m in re.finditer(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)',clean(text)):
  vals.append(float(m.group(1)))
 return max(vals) if vals else 0.0

def extract_education(text):
 t=clean(text); return sorted(set(e.strip() for e in EDU_TERMS if e.strip() in t))

def infer_role(text):
 t=clean(text)
 for role in ROLE_TERMS:
  if role in t:return role
 return 'general technology'

def parse_resume(text):
 return {'skills':extract_skills(text),'years':extract_years(text),'education':extract_education(text),'role':infer_role(text),'raw':text}
parse_job=parse_resume

def protected_flags(text):
 t=clean(text); return sorted(x for x in PROTECTED_TERMS if x in t)

def weighted_match(c,j):
 req=set(j['skills']); have=set(c['skills']); skill=len(req&have)/len(req)*100 if req else 0
 exp=100 if c['years']>=j['years'] else c['years']/max(j['years'],1)*100
 role=100 if c['role']==j['role'] else (55 if c['role']!='general technology' and j['role']!='general technology' else 30)
 edu=100 if set(c['education'])&set(j['education']) else 65
 total=.55*skill+.20*exp+.15*role+.10*edu
 return round(total,2),{'skill':round(skill,2),'experience':round(exp,2),'role':round(role,2),'education':round(edu,2)}

def tfidf_match(c,j):
 vec=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=5000); X=vec.fit_transform([c['raw'],j['raw']])
 return round(float(cosine_similarity(X[0:1],X[1:2])[0,0]*100),2)

def astar_match(c,j):
 req=list(j['skills']); have=set(c['skills'])
 if not req:return 0.0,[],[]
 # State is (requirement_index, matched_tuple). g is unmet cost; h is remaining unmet requirements.
 pq=[(len(req),0,0,tuple())]; seen={}; trace=[]
 while pq:
  f,g,i,matched=heapq.heappop(pq)
  key=(i,matched)
  if key in seen and seen[key]<=g: continue
  seen[key]=g
  if i==len(req): return round(len(matched)/len(req)*100,2),list(matched),trace
  skill=req[i]; hit=skill in have; ng=g+(0 if hit else 1); nm=matched+(skill,) if hit else matched
  rem=req[i+1:]; h=sum(1 for s in rem if s not in have)
  nf=ng+h
  trace.append({'step':i+1,'requirement':skill,'satisfied':hit,'g':ng,'h':h,'f':nf})
  heapq.heappush(pq,(nf,ng,i+1,nm))
 return 0.0,[],trace

def greedy_match(c,j):
 req=list(j['skills']); have=set(c['skills']); matched=[s for s in req if s in have]
 return round(len(matched)/len(req)*100,2) if req else 0.0,matched

def make_features(c,j):
 w,_=weighted_match(c,j); tf=tfidf_match(c,j); ast,_,_=astar_match(c,j)
 _,parts=weighted_match(c,j); return [parts['skill'],parts['experience'],parts['role'],parts['education'],tf,ast,w]

def decision_tree_train(rows):
 X=np.array([make_features(r['candidate'],r['job']) for r in rows]); y=np.array([r['label'] for r in rows])
 clf=DecisionTreeClassifier(max_depth=4,random_state=42); clf.fit(X,y); return clf

def decision_predict(clf,c,j):
 p=clf.predict_proba([make_features(c,j)])[0]; classes=list(clf.classes_); return round(float(p[classes.index(1)])*100,2) if 1 in classes else 0.0

def screen_candidate(c,j,clf=None):
 w,parts=weighted_match(c,j); tf=tfidf_match(c,j); ast,path,trace=astar_match(c,j); greedy,_=greedy_match(c,j)
 final=round(.50*w+.25*tf+.25*ast,2); model=decision_predict(clf,c,j) if clf else None
 matched=sorted(set(j['skills'])&set(c['skills'])); missing=sorted(set(j['skills'])-set(c['skills']))
 reasons=[]
 if matched: reasons.append('Matched core skills: '+', '.join(matched))
 if missing: reasons.append('Skill gaps: '+', '.join(missing))
 reasons.append('A* requirement coverage: %.1f%%.'%ast)
 reasons.append('Greedy coverage: %.1f%%.'%greedy)
 reasons.append('Experience requirement '+('satisfied.' if c['years']>=j['years'] else 'not fully satisfied.'))
 return {'weighted':w,'tfidf':tf,'astar':ast,'greedy':greedy,'final':final,'model':model,'parts':parts,'matched':matched,'missing':missing,'astar_path':path,'astar_trace':trace,'explanation':reasons,'protected_flags_ignored':protected_flags(c['raw'])}

def recommend_jobs(c,jobs,clf=None):
 out=[]
 for j in jobs:
  r=screen_candidate(c,j,clf); out.append({'job_id':j['id'],'job':j['title'],**{k:r[k] for k in ['score'] if k in r},'score':r['final'],'weighted':r['weighted'],'tfidf':r['tfidf'],'astar':r['astar'],'greedy':r['greedy'],'missing':r['missing'],'matched':r['matched'],'explanation':r['explanation']})
 return sorted(out,key=lambda x:x['score'],reverse=True)

def skill_gap_plan(c,j):
 gaps=sorted(set(j['skills'])-set(c['skills']))
 levels=['Foundation','Applied','Project']
 return [{'skill':s,'priority':'High' if i<3 else 'Medium','roadmap':[f'{levels[min(i,2)]} learning',f'Build a {s} mini-project',f'Validate {s} with an interview task']} for i,s in enumerate(gaps)]

def interview_questions(role,skills):
 skills=[s for s in skills if s]
 return [
 f'Explain a production-ready project where you used {skills[0] if skills else "your strongest skill"}. What design trade-off did you make?',
 f'As a {role}, how would you diagnose a failure that appears only under high load?',
 f'You are missing one expected skill. How would you learn and validate it within two weeks?',
 'Describe a technical decision where reliability, cost and performance conflicted. What did you choose?',
 'How would you explain your recommendation to a recruiter who cannot inspect the code?'
 ]
