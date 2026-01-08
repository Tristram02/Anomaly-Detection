import matplotlib
matplotlib.use('Agg')
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from node2vec import Node2Vec
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, auc, f1_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
import json
import os
from collections import Counter

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10



def load_cora_graph(data_dir='cora'):
    print("Wczytywanie grafu Cora")
    
    content_path = os.path.join(data_dir, 'cora.content')
    features = {}
    labels = {}
    
    with open(content_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_id = parts[0]
            feature_vec = np.array([int(x) for x in parts[1:-1]])
            label = parts[-1]
            features[node_id] = feature_vec
            labels[node_id] = label
    
    cites_path = os.path.join(data_dir, 'cora.cites')
    edges = []
    
    with open(cites_path, 'r') as f:
        for line in f:
            cited, citing = line.strip().split()
            if cited in features and citing in features:
                edges.append((citing, cited))
    
    G = nx.DiGraph()
    G.add_nodes_from(features.keys())
    G.add_edges_from(edges)
    
    print(f"Graf wczytany: {G.number_of_nodes()} węzłów, {G.number_of_edges()} krawędzi")
    
    return G, features, labels


def compute_graph_statistics(G):
    print("\n=== Statystyki grafu ===")
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"Liczba węzłów: {n_nodes}")
    print(f"Liczba krawędzi: {n_edges}")
    
    G_undirected = G.to_undirected()
    degrees = dict(G_undirected.degree())
    degree_values = list(degrees.values())
    
    print(f"Średni stopień: {np.mean(degree_values):.2f}")
    print(f"Mediana stopnia: {np.median(degree_values):.2f}")
    print(f"Min stopień: {min(degree_values)}")
    print(f"Max stopień: {max(degree_values)}")
    
    clustering = nx.average_clustering(G_undirected)
    print(f"Średni współczynnik klasteryzacji: {clustering:.4f}")
    
    density = nx.density(G)
    print(f"Gęstość grafu: {density:.6f}")
    
    stats = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'avg_degree': np.mean(degree_values),
        'median_degree': np.median(degree_values),
        'min_degree': min(degree_values),
        'max_degree': max(degree_values),
        'clustering': clustering,
        'density': density,
        'degrees': degrees
    }
    
    return stats


def plot_degree_distribution(degrees, save_path='results/figures/degree_distribution.png'):
    degree_values = list(degrees.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(degree_values, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Stopień węzła')
    ax1.set_ylabel('Liczba węzłów')
    ax1.set_title('Rozkład stopni węzłów')
    ax1.grid(True, alpha=0.3)
    
    degree_count = Counter(degree_values)
    degrees_sorted = sorted(degree_count.keys())
    counts = [degree_count[d] for d in degrees_sorted]
    
    ax2.loglog(degrees_sorted, counts, 'o', alpha=0.7)
    ax2.set_xlabel('Stopień węzła (log)')
    ax2.set_ylabel('Liczba węzłów (log)')
    ax2.set_title('Rozkład stopni (skala log-log)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Zapisano: {save_path}")
    plt.close()


def inject_anomalies(G, anomaly_ratio=0.1, rewiring_ratio=0.5, seed=42):
    np.random.seed(seed)
    G_modified = G.copy()
    
    nodes = list(G.nodes())
    n_anomalies = int(len(nodes) * anomaly_ratio)
    
    anomaly_nodes = np.random.choice(nodes, size=n_anomalies, replace=False)
    
    print(f"\nWstrzykiwanie anomalii: {n_anomalies} węzłów ({anomaly_ratio*100:.1f}%)")
    
    for node in anomaly_nodes:
        out_edges = list(G_modified.out_edges(node))
        in_edges = list(G_modified.in_edges(node))
        
        all_edges = out_edges + in_edges
        
        if len(all_edges) == 0:
            continue
            
        n_rewire = max(1, int(len(all_edges) * rewiring_ratio))
        edges_to_rewire = np.random.choice(len(all_edges), size=min(n_rewire, len(all_edges)), replace=False)
        
        for idx in edges_to_rewire:
            if idx < len(out_edges):
                u, v = out_edges[idx]
                if G_modified.has_edge(u, v):
                    G_modified.remove_edge(u, v)
                    new_target = np.random.choice(nodes)
                    if new_target != u and not G_modified.has_edge(u, new_target):
                        G_modified.add_edge(u, new_target)
            else:
                u, v = in_edges[idx - len(out_edges)]
                if G_modified.has_edge(u, v):
                    G_modified.remove_edge(u, v)
                    new_source = np.random.choice(nodes)
                    if new_source != v and not G_modified.has_edge(new_source, v):
                        G_modified.add_edge(new_source, v)
    
    anomaly_labels = {node: 1 if node in anomaly_nodes else 0 for node in nodes}
    
    print(f"Anomalie wstrzyknięte: {sum(anomaly_labels.values())} węzłów")
    
    return G_modified, anomaly_labels


def generate_node2vec_embeddings(G, dimensions=64, walk_length=80, num_walks=10, 
                                   p=1, q=1, workers=4, seed=42):

    print(f"\nGenerowanie embeddingów Node2Vec (dim={dimensions}, p={p}, q={q})")
    
    node2vec = Node2Vec(
        G, 
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
        quiet=True
    )
    
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    embeddings = {}
    for node in G.nodes():
        embeddings[node] = model.wv[node]
    
    print(f"Embeddingi wygenerowane: {len(embeddings)} węzłów, wymiar {dimensions}")
    
    return embeddings


def detect_anomalies_lof(embeddings, n_neighbors=20, contamination=0.1):
    print(f"\nDetekcja LOF (n_neighbors={n_neighbors})")
    
    nodes = list(embeddings.keys())
    X = np.array([embeddings[node] for node in nodes])
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    lof.fit(X)
    scores = -lof.decision_function(X)
    
    anomaly_scores = {node: score for node, score in zip(nodes, scores)}
    
    return anomaly_scores


def detect_anomalies_if(embeddings, contamination=0.1, random_state=42):
    print(f"\nDetekcja Isolation Forest")
    
    nodes = list(embeddings.keys())
    X = np.array([embeddings[node] for node in nodes])
    
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    iso_forest.fit(X)
    scores = -iso_forest.decision_function(X)
    
    anomaly_scores = {node: score for node, score in zip(nodes, scores)}
    
    return anomaly_scores


def evaluate_detector(anomaly_scores, true_labels):
    nodes = list(anomaly_scores.keys())
    y_true = np.array([true_labels[node] for node in nodes])
    y_scores = np.array([anomaly_scores[node] for node in nodes])
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    roc_auc = roc_auc_score(y_true, y_scores)
    
    threshold = np.percentile(y_scores, 90)
    y_pred = (y_scores >= threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    metrics = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'recall': recall
    }
    
    return metrics


def analyze_anomaly_characteristics(G, anomaly_labels, anomaly_scores):
    print("\n=== Analiza charakterystyk anomalii ===")
    
    G_undirected = G.to_undirected()
    degrees = dict(G_undirected.degree())
    
    print("Obliczanie centralności")
    centrality = nx.betweenness_centrality(G_undirected, k=min(100, G.number_of_nodes()))
    
    normal_nodes = [n for n, label in anomaly_labels.items() if label == 0]
    anomaly_nodes = [n for n, label in anomaly_labels.items() if label == 1]
    
    normal_degrees = [degrees[n] for n in normal_nodes]
    anomaly_degrees = [degrees[n] for n in anomaly_nodes]
    
    print(f"\nŚredni stopień węzłów normalnych: {np.mean(normal_degrees):.2f}")
    print(f"Średni stopień węzłów anomalnych: {np.mean(anomaly_degrees):.2f}")
    
    normal_centrality = [centrality[n] for n in normal_nodes]
    anomaly_centrality = [centrality[n] for n in anomaly_nodes]
    
    print(f"Średnia centralność węzłów normalnych: {np.mean(normal_centrality):.6f}")
    print(f"Średnia centralność węzłów anomalnych: {np.mean(anomaly_centrality):.6f}")
    
    characteristics = {
        'normal_avg_degree': np.mean(normal_degrees),
        'anomaly_avg_degree': np.mean(anomaly_degrees),
        'normal_avg_centrality': np.mean(normal_centrality),
        'anomaly_avg_centrality': np.mean(anomaly_centrality),
        'degrees': degrees,
        'centrality': centrality
    }
    
    return characteristics


def plot_embeddings_2d(embeddings, anomaly_labels, anomaly_scores, 
                        save_path='results/figures/embeddings_2d.png'):

    nodes = list(embeddings.keys())
    X = np.array([embeddings[node] for node in nodes])
    y_true = np.array([anomaly_labels[node] for node in nodes])
    scores = np.array([anomaly_scores[node] for node in nodes])
    
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter1 = ax1.scatter(X_2d[y_true==0, 0], X_2d[y_true==0, 1], 
                           c='blue', alpha=0.5, s=20, label='Normalne')
    scatter2 = ax1.scatter(X_2d[y_true==1, 0], X_2d[y_true==1, 1], 
                           c='red', alpha=0.8, s=30, label='Anomalie', marker='x')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax1.set_title('Embeddingi (prawdziwe etykiety)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    scatter3 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=scores, 
                           cmap='RdYlBu_r', alpha=0.6, s=20)
    plt.colorbar(scatter3, ax=ax2, label='Anomaly Score')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax2.set_title('Embeddingi (wykryte anomalie)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Zapisano: {save_path}")
    plt.close()


def plot_metrics_comparison(results, save_path='results/figures/metrics_comparison.png'):

    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['pr_auc', 'roc_auc', 'f1_score', 'recall']
    titles = ['PR-AUC', 'ROC-AUC', 'F1-Score', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for detector in df['detector'].unique():
            df_det = df[df['detector'] == detector]
            x_labels = [f"p={row['p']}, q={row['q']}, d={row['dim']}" 
                       for _, row in df_det.iterrows()]
            ax.plot(range(len(df_det)), df_det[metric], marker='o', label=detector)
        
        ax.set_xlabel('Konfiguracja')
        ax.set_ylabel(title)
        ax.set_title(f'Porównanie: {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Zapisano: {save_path}")
    plt.close()


def main():
    G, features, labels = load_cora_graph()
    
    stats = compute_graph_statistics(G)
    plot_degree_distribution(stats['degrees'])
    
    G_modified, anomaly_labels = inject_anomalies(G, anomaly_ratio=0.1, rewiring_ratio=0.5)
    
    results = []
    
    dimensions_list = [32, 64, 128]
    p_values = [0.5, 1, 2]
    q_values = [0.5, 1, 2]
    
    configs = [
        {'dim': 64, 'p': 1, 'q': 1},
        {'dim': 64, 'p': 0.5, 'q': 2},
        {'dim': 64, 'p': 2, 'q': 0.5},
        {'dim': 32, 'p': 1, 'q': 1},
        {'dim': 128, 'p': 1, 'q': 1},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Konfiguracja: dim={config['dim']}, p={config['p']}, q={config['q']}")
        print(f"{'='*60}")

        embeddings = generate_node2vec_embeddings(
            G_modified, 
            dimensions=config['dim'],
            p=config['p'],
            q=config['q']
        )
        
        print("\n--- LOF ---")
        lof_scores = detect_anomalies_lof(embeddings)
        lof_metrics = evaluate_detector(lof_scores, anomaly_labels)
        
        results.append({
            'detector': 'LOF',
            'dim': config['dim'],
            'p': config['p'],
            'q': config['q'],
            **lof_metrics
        })
        
        print("\n--- Isolation Forest ---")
        if_scores = detect_anomalies_if(embeddings)
        if_metrics = evaluate_detector(if_scores, anomaly_labels)
        
        results.append({
            'detector': 'IsolationForest',
            'dim': config['dim'],
            'p': config['p'],
            'q': config['q'],
            **if_metrics
        })
        
        if config == configs[0]:
            plot_embeddings_2d(embeddings, anomaly_labels, lof_scores, 
                              'results/figures/embeddings_lof.png')
            plot_embeddings_2d(embeddings, anomaly_labels, if_scores,
                              'results/figures/embeddings_if.png')
    
    embeddings_baseline = generate_node2vec_embeddings(G_modified, dimensions=64, p=1, q=1)
    characteristics = analyze_anomaly_characteristics(G_modified, anomaly_labels, 
                                                     detect_anomalies_lof(embeddings_baseline))
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/metrics.csv', index=False)
    print("\nWyniki zapisane do results/metrics.csv")
    
    with open('results/metrics.json', 'w') as f:
        json.dump({
            'graph_stats': {k: v for k, v in stats.items() if k != 'degrees'},
            'characteristics': {k: v for k, v in characteristics.items() 
                              if k not in ['degrees', 'centrality']},
            'results': results
        }, f, indent=2)
    
    plot_metrics_comparison(results)

if __name__ == '__main__':
    main()
