# Autonomous Combat Identification Engine
**Tactical Reasoning Engine** implementing the **ID3** algorithm from scratch to automate **Combat Identification (CID)**.

## Technical Description
* **Architecture:** Non-parametric hierarchical classifier designed for **sensor fusion** and explainable decision-making.
* **Data Flow:**
  * **Source:** Synthetic tactical dataset with 10k detections, simulating a high-saturation air combat environment (e.g., Iran). *Generated via `python dataset/script_generator.py`.*
  * **Dimensions:** Symbolic feature vectors: IFF Response, Radar Signature/RCS, Flight Profile and BFT Proximity.
  * **Target:** Binary classification (1 Friend / 0 Foe) for weapon release authorization.
* **Objective:** Minimize **Shannon Entropy** through **Information Gain** to achieve robust identification, even during primary transponder failure, preventing blue-on-blue incidents.

## Project Organization
*   `dataset/`: Contains the synthetic generator and the tactical dataset (10k samples).
*   `scripts/functions.py`: The core engine. Implements Shannon Entropy, Information Gain, and the ID3 recursive builder.
*   `test/`: Tools for generating independent test cases (1k samples) and validation.
*   `comparison/`: Scripts to contrast the Engine's performance against baseline sensors (IFF-only).

## Engine Architecture (Fast-ID3)
To guarantee ultra-fast execution in high-saturation combat scenarios, the engine's architecture is decoupled into three specialized functions centered around a nested dictionary data structure:

1.  **Attribute Prioritization (`sort_attributes`)**: First, the engine calculates the Conditional Shannon Entropy for each attribute across the dataset. This establishes a fixed tactical hierarchy indicating which sensors provide the most information gain.
2.  **Tree Construction (`build_tree`)**: Using the sorted hierarchy, the system recursively segments the training data to build the ID3 decision tree. Crucially, this tree is instantiated as a **nested dictionary** in memory. Each key represents a specific sensor reading (like `{'Valida': ...}`), and its value holds either the next sub-dictionary (the next sensor to check) or a final classification (1 for Friend, 0 for Foe).
3.  **Inference (`predict_case`)**: When evaluating a live target, the engine simply queries this root dictionary using the target's parameters. Instead of filtering data matrices or running complex "if-else" cascades, Python transverses the nested hash maps. Determining the classification is thus reduced to an instant sequence of memory lookups, optimizing execution speed to the maximum.

## Conclusion: Improvement
By evolving from a single-sensor logic (**IFF-only**) to a multi-attribute decision tree (**ID3 Engine**), we increased identification reliability from **94.6% to 99.5%**. 

The **99.5% result represents the theoretical maximum** for this dataset. Due to the intentional noise introduced during data generation (simulating sensor read failures), there are instances with identical attributes but conflicting classifications. This makes any further improvement mathematically impossible without introducing additional sensors or higher-resolution data.


## Dependencies installation
Write these commands one by one in the terminal of your editor to install the dependencies:

**Mac / Linux:**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
