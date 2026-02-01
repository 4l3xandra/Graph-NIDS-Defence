import pandas as pd
import numpy as np
import networkx as nx
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# CONFIGURATION 
# The file is expected to be in the same directory as this script
DATA_PATH = "Wednesday-workingHours.pcap_ISCX.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
tf.random.set_seed(42)
np.random.seed(42)

# ------------------------------------------
# 1. ETL & GRAPH FEATURE ENGINEERING
# ------------------------------------------
def load_and_clean_data(path):
    if not os.path.exists(path):
        print(f"[!] Error: Dataset not found at {path}")
        print("Please ensure you have the CSV file in the same path as this code.")
        sys.exit(1)
    
    print(f"[*] Loading dataset: {os.path.basename(path)}...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    if 'Label' in df.columns:
        df['binary_label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    else:
        raise ValueError("Column 'Label' not found.")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"    Final Shape: {df.shape}")
    return df

def extract_graph_features(df_subset):
    """ Extracts PageRank and Degree Centrality """
    G = nx.from_pandas_edgelist(
        df_subset, source='Source IP', target='Destination IP', 
        create_using=nx.DiGraph() 
    )
    
    pagerank = pd.Series(nx.pagerank(G, alpha=0.85), name='pagerank')
    in_degree = pd.Series(dict(G.in_degree()), name='log_in_degree').apply(np.log1p)
    out_degree = pd.Series(dict(G.out_degree()), name='log_out_degree').apply(np.log1p)
    
    ip_metrics = pd.concat([pagerank, in_degree, out_degree], axis=1).reset_index().rename(columns={'index': 'IP'})
    
    # Merge Source
    df_out = df_subset.merge(ip_metrics, left_on='Source IP', right_on='IP', how='left')
    df_out = df_out.rename(columns={'pagerank': 'src_pagerank', 'log_in_degree': 'src_in_degree', 'log_out_degree': 'src_out_degree'}).drop(columns=['IP'])

    # Merge Destination
    df_out = df_out.merge(ip_metrics, left_on='Destination IP', right_on='IP', how='left')
    df_out = df_out.rename(columns={'pagerank': 'dst_pagerank', 'log_in_degree': 'dst_in_degree', 'log_out_degree': 'dst_out_degree'}).drop(columns=['IP'])
    
    return df_out.fillna(0)

def process_pipeline():
    df = load_and_clean_data(DATA_PATH)
    
    # Split before Graph Gen to prevent Leakage
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['binary_label'])

    print("[*] Extracting Graph Features...")
    train_df_graph = extract_graph_features(train_df)
    test_df_graph = extract_graph_features(test_df)

    drop_cols = ['Source IP', 'Destination IP', 'Timestamp', 'Flow ID', 'Label', 'binary_label']
    graph_cols = ['src_pagerank', 'src_in_degree', 'src_out_degree', 'dst_pagerank', 'dst_in_degree', 'dst_out_degree']

    y_train = train_df_graph['binary_label'].values
    y_test = test_df_graph['binary_label'].values

    # Full Feature Set (Graph + Stat)
    X_train_full = train_df_graph.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
    X_test_full = test_df_graph.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])

    # Stat Only Set
    X_train_stat_raw = X_train_full.drop(columns=graph_cols, errors='ignore')
    X_test_stat_raw = X_test_full.drop(columns=graph_cols, errors='ignore')

    print("[*] Scaling features...")
    scaler_stat = StandardScaler()
    X_train_stat = scaler_stat.fit_transform(X_train_stat_raw).astype('float32')
    X_test_stat = scaler_stat.transform(X_test_stat_raw).astype('float32')

    scaler_graph = StandardScaler()
    X_train_graph = scaler_graph.fit_transform(X_train_full).astype('float32')
    X_test_graph = scaler_graph.transform(X_test_full).astype('float32')
    
    # Return test_df to look up Attack Names later
    return X_train_stat, X_test_stat, X_train_graph, X_test_graph, y_train, y_test, test_df
