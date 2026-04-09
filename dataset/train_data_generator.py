import pandas as pd
import numpy as np
import random
import os

def generate_tactical_dataset(n_samples=10000):
    data = []
    
    # Object types and their logical profiles
    # Format: (Name, Is_Friend[1/0], Prob_IFF_Valid, Expected_RCS)
    profiles = [
        ('Allied_Fighter', 1, 0.95, 'Medium'),
        ('Allied_Stealth_Fighter', 1, 0.90, 'Small'),
        ('Commercial', 1, 0.99, 'Large'),
        ('Allied_Drone', 1, 0.85, 'Small'),
        ('Enemy_Fighter', 0, 0.05, 'Medium'),
        ('Enemy_Stealth_Fighter', 0, 0.01, 'Very_Small'),
        ('Enemy_Drone', 0, 0.02, 'Small'),
        ('Cruise_Missile', 0, 0.01, 'Small')
    ]

    for _ in range(n_samples):
        # Select a random profile
        name, is_friend, p_iff, rcs_base = random.choice(profiles)
        
        # 1. IFF Response (Simulating failures)
        iff_rand = random.random()
        if iff_rand < p_iff:
            iff = 'Valid'
        elif iff_rand < p_iff + 0.05:
            iff = 'Invalid' # Code error
        else:
            iff = 'Absent' # Not responding or turned off

        # 2. Flight Profile
        if is_friend:
            profile = np.random.choice(['Correct', 'Deviated'], p=[0.8, 0.2])
        else:
            profile = np.random.choice(['Erratic', 'Deviated'], p=[0.7, 0.3])

        # 3. Radar Signature (RCS) - With noise to fool the tree
        rcs = rcs_base
        if random.random() < 0.1: # 10% reading error
            rcs = random.choice(['Very_Small', 'Small', 'Medium', 'Large'])

        # 4. Allied Proximity (in Km)
        if is_friend:
            prox = random.uniform(0, 50)
        else:
            prox = random.uniform(30, 200)
        
        # Discretize proximity for ID3 (Close, Medium, Far)
        if prox < 20: prox_cat = 'Close'
        elif prox < 80: prox_cat = 'Medium'
        else: prox_cat = 'Far'

        data.append([rcs, iff, profile, prox_cat, is_friend])

    columns = ['Radar_Signature', 'IFF_Response', 'Flight_Profile', 'Allied_Proximity', 'Target_Friend']
    df = pd.DataFrame(data, columns=columns)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'dataset.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated with {n_samples} records.")

if __name__ == "__main__":
    generate_tactical_dataset()