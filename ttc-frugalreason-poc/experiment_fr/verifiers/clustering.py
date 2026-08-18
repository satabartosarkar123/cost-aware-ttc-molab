import re
from typing import List, Tuple, Dict
from collections import Counter

def rationale_bigrams(text: str) -> set:
    """Extracts alphanumeric bigrams from lowercased text."""
    if not text:
        return set()
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 2:
        return set(words)
    return set(zip(words[:-1], words[1:]))

def jaccard(a: set, b: set) -> float:
    """Calculates Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union > 0 else 0.0

def cluster_rationales(rationales: List[str], threshold: float = 0.7) -> List[int]:
    """
    Groups rationales into clusters using a union-find approach.
    Two samples merge if their bigram jaccard similarity > threshold.
    Returns a list of cluster IDs for each rationale.
    """
    n = len(rationales)
    parent = list(range(n))
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            
    sets = [rationale_bigrams(r) for r in rationales]
    
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(sets[i], sets[j]) > threshold:
                union(i, j)
                
    # Normalize cluster IDs
    cluster_ids = [find(i) for i in range(n)]
    unique_ids = {}
    normalized = []
    for cid in cluster_ids:
        if cid not in unique_ids:
            unique_ids[cid] = len(unique_ids)
        normalized.append(unique_ids[cid])
        
    return normalized

def cluster_weighted_vote(rationales: List[str], answers: List[str], threshold: float = 0.7) -> Tuple[str, float, Dict]:
    """
    Performs cluster-weighted voting. 
    Each CLUSTER contributes 1 vote to its majority answer (not each sample).
    Returns (winner, consistency_c, cluster_structure).
    """
    if not rationales:
        return None, 0.0, {}
        
    clusters = cluster_rationales(rationales, threshold)
    
    # Map cluster ID to list of (index, rationale, answer)
    cluster_map = {}
    for idx, (cid, r, a) in enumerate(zip(clusters, rationales, answers)):
        if cid not in cluster_map:
            cluster_map[cid] = []
        cluster_map[cid].append({"index": idx, "rationale": r, "answer": a})
        
    cluster_votes = Counter()
    for cid, members in cluster_map.items():
        # Find majority answer within this cluster
        ans_counts = Counter(m["answer"] for m in members if m["answer"] is not None)
        if ans_counts:
            cluster_majority = ans_counts.most_common(1)[0][0]
            cluster_votes[cluster_majority] += 1
            
    if not cluster_votes:
        return None, 0.0, cluster_map
        
    winner = cluster_votes.most_common(1)[0][0]
    votes_for_winner = cluster_votes[winner]
    total_clusters = len(cluster_map)
    consistency_c = votes_for_winner / total_clusters if total_clusters > 0 else 0.0
    
    return winner, consistency_c, cluster_map
