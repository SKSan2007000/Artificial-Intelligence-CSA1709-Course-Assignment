
import sys, os, itertools, json, time
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"app"))
from engine import *
from data import CANDIDATES,JOBS

cp=[{**c,**parse_resume(c["text"])} for c in CANDIDATES]
jp=[{**j,**parse_job(j["text"])} for j in JOBS]

rows=[]
for c in cp:
    for j in jp:
        s,_=weighted_match(c,j)
        rows.append({"candidate":c,"job":j,"label":1 if s>=65 else 0})
clf=decision_tree_train(rows)

records=[]
for c in cp:
    true=[]
    methods={"Weighted":[],"TF-IDF":[],"A* Hybrid":[]}
    for j in jp:
        w,_=weighted_match(c,j); tf=tfidf_match(c,j); ast,_,_=astar_match(c,j)
        final=.5*w+.25*tf+.25*ast
        true.append(1 if w>=65 else 0)
        methods["Weighted"].append(w); methods["TF-IDF"].append(tf); methods["A* Hybrid"].append(final)
    for name,scores in methods.items():
        order=np.argsort(scores)[::-1]
        k=3
        p=sum(true[i] for i in order[:k])/k
        rel=np.array([[true[i] for i in order[:k]]])
        nd=float(ndcg_score(rel, np.array([[scores[i] for i in order[:k]]])))
        records.append({"candidate":c["id"],"method":name,"precision_at_3":p,"ndcg_at_3":nd})
df=pd.DataFrame(records)
summary=df.groupby("method")[["precision_at_3","ndcg_at_3"]].mean().reset_index()
summary.to_csv(os.path.join(os.path.dirname(__file__),"benchmark.csv"),index=False)

# decision tree classification metrics on held-out synthetic pairs
X=np.array([make_features(r["candidate"],r["job"]) for r in rows]); y=np.array([r["label"] for r in rows])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
clf2=DecisionTreeClassifier(max_depth=4,random_state=42).fit(Xtr,ytr)
pred=clf2.predict(Xte)
metrics={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0)}
with open(os.path.join(os.path.dirname(__file__),"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
print(summary)
print(metrics)
