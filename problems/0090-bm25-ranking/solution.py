import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
	# STATS 
	N = len(corpus); avgdl = sum(len(d) for d in corpus) / N 
	# DF 
	df = Counter()
	for d in corpus:
		for term in set(d): df[term] += 1
	scores = []
	for d in corpus:
		sc = 0; tfc = Counter(d)
		for q in query:
			if q not in tfc: continue 
			# IDF 
			idf = np.log((N + 1) / (df[q] + 1))
			tf = tfc[q]
			# SATURATE 
			sc += idf * (tf*(k1 + 1))/(tf + k1 * (1 - b + b * len(d)/avgdl))
		# SCORE 
		scores.append(sc)
	return [round(sc, 3) for sc in scores]
	
	