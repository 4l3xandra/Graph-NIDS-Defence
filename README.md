# Graph-NIDS-Defence
A robust Network Intrusion Detection System using topological graph features (PageRank, Degree Centrality) to defend against adversarial evasion attacks.

## Overview
-**Dataset:** CIC-IDS2017 (Wednesday - DoS Attacks).
-**Method:** Hybrid Deep Learning (MLP) combining Statistical + Graph Features.
-**Defense:** Adversarial Training & Topological Feature Locking.
-**Key Result:** Restores detection accuracy from ~0% (under attack) to >95% (with graph defense).

## Dataset
This project uses the **CICIDS2017** dataset, specifically the `Wednesday-workingHours.pcap_ISCX.csv` (DoS Attacks).
The dataset is not included in this repository.

*Download Instructions:*
1. Download `Wednesday-workingHours.pcap_ISCX.csv` from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Place the CSV file in the **root directory** (same folder as `Graph-NIDS-Defence.py`).

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/4l3xandra/Graph-NIDS-Defence.git](https://github.com/4l3xandra/Graph-NIDS-Defence.git)
   cd Graph-NIDS-Defence
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
## Run the File
```bash
   python Graph-NIDS-Defence
