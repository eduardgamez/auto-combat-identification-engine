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
The identifying engine is optimized for high-saturation environments using a two-phase approach:

1.  **Build Phase (Training)**: The `build_tree` function transforms the raw tactical data into a **Nested Dictionary Tree** in memory. It uses recursion to find the optimal split at each level based on conditional entropy. 
2.  **Execution Phase (Inference)**: Unlike standard ID3 implementations that might re-filter data, our `predict_case` function performs a **Direct Memory Hash Lookup**. Classification time is `O(d)`, where `d` is the number of attributes, making it capable of processing thousands of detections per second.

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
