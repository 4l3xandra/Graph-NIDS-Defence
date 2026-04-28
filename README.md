# Graph-NIDS-Defence
A robust Network Intrusion Detection System using topological graph features (PageRank, Degree Centrality) to defend against adversarial evasion attacks.

## Overview
- **Dataset:** CIC-IDS2017.
- **Method:** Hybrid Deep Learning (MLP) combining Statistical + Graph Features.
- **Defence:** Adversarial Training & Topological Feature Locking.
- **Key Result:** Restores detection accuracy from ~65% (under attack) to >98% (with graph defence).

## Dataset
This project uses the **CICIDS2017** dataset, specifically the `Wednesday-workingHours.pcap_ISCX.csv` (DoS Attacks).
The dataset is not included in this repository.

*Download Instructions:*
1. Download `Wednesday-workingHours.pcap_ISCX.csv` from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html).
2. The script expects this CSV file to be placed in the root directory of this project.
3. (Optional) If you want to store the dataset elsewhere, you can point the script to your folder path using the --data argument in the command line.

## Installation
Python 3.11 is recommended for compatibility with TensorFlow.
1. Clone the repository:
   ```bash
   git clone https://github.com/4l3xandra/Graph-NIDS-Defence.git
   cd Graph-NIDS-Defence
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
## Run the File
- For default configuration:
```bash
   python Graph-NIDS-Defence
```
(Note: The script will automatically generate .png visualizations of the Confusion Matrix, Detection Rates, Robustness Comparison, and Feature Importances.)
- For advanced configuration:

The project features a Command-Line Interface (CLI) allowing modification of the pipeline parameters.

**View all commands:**

```bash
python Graph-NIDS-Defence.py --help
```

**Custom execution example:**

For 5,000 samples and a 70/30 split:
```bash
python Graph-NIDS-Defence.py --samples 5000 --test_size 0.3
```

**Available arguments:**

--data : Path to the dataset CSV file (default: Wednesday-workingHours.pcap_ISCX.csv). (Note that custom datasets must contain 'Source IP', 'Destination IP', 'Timestamp', and 'Label' columns to generate the graph topologies.)

--samples : Total number of test-set network flows to subject to the FGSM stress test (default: 10000).

--test_size : Stratified chronological split ratio (default: 0.2).

--seed : Mathematical seed for keeping the train/test splits and network weights consistent across different runs (default: 42).

--epochs EPOCHS       Number of training epochs for the neural networks (default: 5).

--epsilon EPSILON     Perturbation magnitude for the FGSM attack (default: 0.1).
