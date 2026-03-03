import pandas as pd
import numpy as np
import networkx as nx
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
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
        sys.exit(1)
    
    print(f"[*] Loading dataset: {os.path.basename(path)}...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    if 'Label' in df.columns:
        df['binary_label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    else:
        raise ValueError("Column 'Label' not found.")

    # Convert and sort by Timestamp for windowing
    print("Parsing timestamps...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp')
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Final Shape: {df.shape}")
    return df

def extract_windowed_graph_features(df, window_size='5min'):
    print(f"[*] Extracting graph features using {window_size} rolling windows...")
    processed_chunks = []
    
    # Group the data into discrete time windows
    for time_idx, group in df.groupby(pd.Grouper(key='Timestamp', freq=window_size)):
        if len(group) == 0:
            continue
            
        # Build graph only for this 5-minute slice
        G = nx.from_pandas_edgelist(
            group, source='Source IP', target='Destination IP', 
            create_using=nx.DiGraph() 
        )
        
        # Calculate metrics
        try:
            pagerank_dict = nx.pagerank(G, alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            # Provide a uniform probability fallback instead of zeros
            node_count = len(G.nodes())
            uniform_pr = 1.0 / node_count if node_count > 0 else 0
            pagerank_dict = {node: uniform_pr for node in G.nodes()} 
            
        in_degree_dict = {node: np.log1p(deg) for node, deg in G.in_degree()}
        out_degree_dict = {node: np.log1p(deg) for node, deg in G.out_degree()}
        
        # Map back to this specific time slice
        group_out = group.copy()
        group_out['src_pagerank'] = group_out['Source IP'].map(pagerank_dict).fillna(0)
        group_out['src_in_degree'] = group_out['Source IP'].map(in_degree_dict).fillna(0)
        group_out['src_out_degree'] = group_out['Source IP'].map(out_degree_dict).fillna(0)
        
        group_out['dst_pagerank'] = group_out['Destination IP'].map(pagerank_dict).fillna(0)
        group_out['dst_in_degree'] = group_out['Destination IP'].map(in_degree_dict).fillna(0)
        group_out['dst_out_degree'] = group_out['Destination IP'].map(out_degree_dict).fillna(0)
        
        processed_chunks.append(group_out)
        
    print("[*] Windowed feature extraction complete.")
    return pd.concat(processed_chunks).reset_index(drop=True)

def process_pipeline(data_path):
    df = load_and_clean_data(data_path)
    
    # 1. Extract dynamic windowed graph features
    df_graph = extract_windowed_graph_features(df, window_size='5min')
    
    # 2. Split dataset chronologically to prevent time-series data leakage
    print("[*] Splitting dataset chronologically...")
    df_graph['window_id'] = df_graph['Timestamp'].dt.floor('5min')
    
    unique_windows = df_graph['window_id'].sort_values().unique()
    split_point = int(len(unique_windows) * (1 - TEST_SIZE))
    train_windows = unique_windows[:split_point]

    train_df_graph = df_graph[df_graph['window_id'].isin(train_windows)].copy()
    test_df_graph = df_graph[~df_graph['window_id'].isin(train_windows)].copy()

    # 3. Extract labels after the final split
    y_train = train_df_graph['binary_label'].values
    y_test = test_df_graph['binary_label'].values
    test_df = test_df_graph.copy() # Save for calibration later

    drop_cols = ['Source IP', 'Destination IP', 'Timestamp', 'Flow ID', 'Label', 'binary_label', 'window_id']
    graph_cols = ['src_pagerank', 'src_in_degree', 'src_out_degree', 'dst_pagerank', 'dst_in_degree', 'dst_out_degree']

    # Enforce column order to guarantee graph_cols are strictly at the end for the masking logic
    stat_cols = [c for c in train_df_graph.columns if c not in drop_cols and c not in graph_cols]
    ordered_cols = stat_cols + graph_cols

    X_train_full = train_df_graph[ordered_cols].select_dtypes(include=[np.number])
    X_test_full = test_df_graph[ordered_cols].select_dtypes(include=[np.number])

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

    feature_names = X_train_full.columns.tolist()
    return X_train_stat, X_test_stat, X_train_graph, X_test_graph, y_train, y_test, test_df, feature_names, len(graph_cols)

# ------------------------------------------
# 2. MODEL UTILITIES
# ------------------------------------------
def build_and_train_model(X_train, y_train, input_dim, model_name="Model"):
    print(f"\n[{model_name}] Building Model (Input: {input_dim})...")
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
    return model

def run_constrained_fgsm(model, X_input, y_input, mask, epsilon):
    """ Generates White-Box Adversarial Examples (Gradient from Target Model) """
    with tf.GradientTape() as tape:
        tape.watch(X_input)
        pred = model(X_input, training=False)
        loss = tf.keras.losses.binary_crossentropy(y_input, pred)
    
    gradient = tape.gradient(loss, X_input)
    constrained_grad = gradient * mask 
    X_adv = (X_input + epsilon * tf.sign(constrained_grad)).numpy()
    return X_adv
    
    # Return test_df to look up Attack Names later
    return X_train_stat, X_test_stat, X_train_graph, X_test_graph, y_train, y_test, test_df

# ------------------------------------------
# 3. MAIN EXECUTION FLOW
# ------------------------------------------
def main(args):
    # A. Load data
    X_train_stat, X_test_stat, X_train_graph, X_test_graph, y_train, y_test, test_df, feature_names, graph_cols_count = process_pipeline(args.data)

    # B. Train Baseline (Statistical)
    model_baseline = build_and_train_model(X_train_stat, y_train, X_train_stat.shape[1], "Baseline")
    _, clean_acc = model_baseline.evaluate(X_test_stat, y_test, verbose=0)
    
    print("[*] Generating Baseline Attack Data...")
    sample_size = min(2000, len(X_test_stat))
    _, atk_acc = model_baseline.evaluate(
        run_constrained_fgsm(model_baseline, tf.convert_to_tensor(X_test_stat[:sample_size]), 
                             tf.convert_to_tensor(y_test[:sample_size].reshape(-1,1).astype('float32')), 
                             None, 0.1),
        y_test[:sample_size], verbose=0
    )

    # C. Train Graph Model
    model_graph = build_and_train_model(X_train_graph, y_train, X_train_graph.shape[1], "Graph-Enhanced")

    # D. Attack simulation (Stress test)
    print("\n" + "="*70)
    print("STRESS TEST: Constrained Attacks (Graph Features Locked)")
    print(f"{'EPSILON':<10} | {'ATTACK TYPE':<15} | {'ACCURACY':<10} | {'RESULT'}")
    print("="*70)
    
    mask_array = np.ones(X_test_graph.shape[1], dtype=np.float32)
    mask_array[-graph_cols_count:] = 0.0 
    mask_tensor = tf.convert_to_tensor(mask_array)

    indices = np.random.choice(len(X_test_graph), size=sample_size, replace=False)
    X_eval = tf.convert_to_tensor(X_test_graph[indices], dtype=tf.float32)
    y_eval = tf.convert_to_tensor(np.expand_dims(y_test[indices], -1), dtype=tf.float32)

    acc_w_final, acc_b_final = 0, 0

    for epsilon in [0.1, 0.3]:
        # 1. White-Box Attack
        X_white = run_constrained_fgsm(model_graph, X_eval, y_eval, mask_tensor, epsilon)
        _, acc_w = model_graph.evaluate(X_white, y_eval.numpy(), verbose=0)
        status_w = "RESISTED" if acc_w > 0.9 else "FAILED"
        print(f"{epsilon:<10} | {'White-Box':<15} | {acc_w:.4f}     | {status_w}")
        if epsilon == 0.1: acc_w_final = acc_w

        # 2. Black-Box Attack
        X_eval_stat_only = X_eval[:, :-graph_cols_count]
        with tf.GradientTape() as tape:
            tape.watch(X_eval_stat_only)
            pred = model_baseline(X_eval_stat_only, training=False)
            loss = tf.keras.losses.binary_crossentropy(y_eval, pred)
        
        grad_stat = tape.gradient(loss, X_eval_stat_only)
        # Use tf.shape() for dynamic extraction to guarantee compatibility with graph execution
        zeros_padding = tf.zeros((tf.shape(grad_stat)[0], graph_cols_count), dtype=tf.float32)
        full_grad = tf.concat([grad_stat, zeros_padding], axis=1)
        
        X_black = (X_eval + epsilon * tf.sign(full_grad * mask_tensor)).numpy()
        _, acc_b = model_graph.evaluate(X_black, y_eval.numpy(), verbose=0)
        status_b = "RESISTED" if acc_b > 0.9 else "FAILED"
        print(f"{epsilon:<10} | {'Black-Box':<15} | {acc_b:.4f}     | {status_b}")
        if epsilon == 0.1: acc_b_final = acc_b
        print("-" * 70)
