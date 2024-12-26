# enhanced_insulin_calculator.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
from typing import Dict, Tuple, List
import seaborn as sns

# -------------
# Medical Disclaimer
# -------------
DISCLAIMER = """
**Medical Disclaimer**:
- This application is for **educational/demo** purposes only.
- It should **NOT** be used as a substitute for professional medical advice.
- Always consult your healthcare provider before making changes to your insulin regimen.
"""

# -------------
# Pharmacology Class
# -------------
class DiabetesPharmacology:
    """Pharmacokinetic/Pharmacodynamic parameters for diabetes medications."""
    def __init__(self):
        # NovoRapid (Insulin Aspart) PK/PD
        self.novorapid_params = {
            'onset_minutes': 15,
            'peak_hours': 1.5,
            'duration_hours': 4,
            'half_life_minutes': 81,
            'distribution_volume': 0.142,  # L/kg
            'clearance_rate': 1.5  # L/hour
        }
        
        # Tresiba (Insulin Degludec) PK/PD
        self.tresiba_params = {
            'onset_hours': 1,
            'steady_state_days': 3,
            'duration_hours': 42,
            'half_life_hours': 25,
            'distribution_volume': 0.307,  # L/kg
            'peak_less_profile': True
        }
        
        # Janumet (Sitagliptin + Metformin) PK/PD
        self.janumet_params = {
            'sitagliptin_half_life_hours': 12.4,
            'metformin_half_life_hours': 6.2,
            'dpp4_inhibition_duration': 24,  # hours
            'peak_effect_hours': 4,
            'bioavailability': 0.87
        }
        
        # Jardiance (Empagliflozin) PK/PD
        self.jardiance_params = {
            'half_life_hours': 12.4,
            'time_to_peak_hours': 1.5,
            'glucose_excretion_peak_hours': 3,
            'steady_state_days': 6,
            'bioavailability': 0.88
        }

# -------------
# Food Database
# -------------
class SouthIndianFoodDatabase:
    """Database of South Indian foods with carbohydrate content, GI, protein, fat."""
    def __init__(self):
        self.foods = {
            'rice': {
                'carbs_per_cup': 45,
                'gi': 73,
                'protein_g': 4.2,
                'fat_g': 0.4,
                'fiber_g': 0.6
            },
            'idli': {
                'carbs_per_piece': 12,
                'gi': 68,
                'protein_g': 2.0,
                'fat_g': 0.1,
                'fiber_g': 0.8
            },
            'dosa': {
                'carbs_per_piece': 30,
                'gi': 69,
                'protein_g': 3.5,
                'fat_g': 1.5,
                'fiber_g': 1.2
            },
            'sambar': {
                'carbs_per_cup': 15,
                'gi': 45,
                'protein_g': 3.0,
                'fat_g': 2.0,
                'fiber_g': 4.5
            },
            'upma': {
                'carbs_per_cup': 35,
                'gi': 65,
                'protein_g': 3.5,
                'fat_g': 4.0,
                'fiber_g': 2.0
            },
            'vada': {
                'carbs_per_piece': 18,
                'gi': 72,
                'protein_g': 4.5,
                'fat_g': 9.0,
                'fiber_g': 1.5
            },
            'coffee_with_milk': {
                'carbs_per_cup': 12,
                'gi': 35,
                'protein_g': 3.3,
                'fat_g': 3.2,
                'fiber_g': 0
            },
            'ice_cream': {
                'carbs_per_cup': 32,
                'gi': 61,
                'protein_g': 3.5,
                'fat_g': 11.0,
                'fiber_g': 0.7
            }
        }

# -------------
# Insulin Calculator
# -------------
class InsulinCalculator:
    def __init__(self, patient_weight: float = 92.0):
        self.patient_weight = patient_weight
        self.pharmacology = DiabetesPharmacology()
        self.food_db = SouthIndianFoodDatabase()
        
        # Patient-specific parameters
        self.current_tresiba = 25  # units
        self.target_bg = 140       # mg/dL
        self.bg_correction_threshold = 180  # mg/dL
        
        # Approx. daily insulin usage:
        #   25u Tresiba + ~50u NovoRapid => total ~75u
        #
        # Include the medication synergy:
        self.janumet_effect = 0.85   # ~15% insulin-sparing
        self.jardiance_effect = 0.90 # ~10% insulin-sparing
        
        # Now define ICR, ISF
        self.icr = self._calculate_icr()
        self.isf = self._calculate_isf()

    def _calculate_icr(self) -> float:
        """Calculate insulin-to-carb ratio, factoring in medication synergy."""
        # Base formula (500 rule): 500 / TDD
        base_icr = 500 / (self.current_tresiba + 50)  # 75 total
        adjusted_icr = base_icr * self.janumet_effect * self.jardiance_effect
        return round(adjusted_icr, 1)
    
    def _calculate_isf(self) -> float:
        """Calculate insulin sensitivity (1800 rule), factoring in synergy."""
        base_isf = 1800 / (self.current_tresiba + 50)
        adjusted_isf = base_isf * self.janumet_effect * self.jardiance_effect
        return round(adjusted_isf, 1)

    def calculate_meal_insulin(self, 
                               current_bg: float,
                               meal_contents: Dict[str, float],
                               meal_time: str) -> Tuple[float, Dict, pd.DataFrame]:
        """
        Calculate mealtime insulin dose and predict a 6-hour glucose curve.
        
        Args:
            current_bg (float): Current blood glucose (mg/dL).
            meal_contents (dict): Food items + quantities.
            meal_time (str): 'breakfast', 'lunch', or 'dinner'.
        
        Returns:
            final_dose (float): Recommended insulin dose (units).
            details (dict): Breakdown of calculations.
            prediction_df (pd.DataFrame): 6-hour predicted BG data.
        """
        total_carbs = 0.0
        weighted_gi = 0.0
        total_protein = 0.0
        total_fat = 0.0
        
        # Summation of carbs/protein/fat
        for food, qty in meal_contents.items():
            fd = self.food_db.foods.get(food, None)
            if not fd:
                continue
            
            if 'carbs_per_cup' in fd:
                carbs = fd['carbs_per_cup'] * qty
            else:
                carbs = fd['carbs_per_piece'] * qty
            
            weighted_gi += fd['gi'] * carbs
            total_carbs += carbs
            total_protein += fd['protein_g'] * qty
            total_fat += fd['fat_g'] * qty
        
        if total_carbs > 0:
            weighted_gi /= total_carbs
        
        # Carb coverage
        carb_dose = total_carbs / self.icr
        
        # Correction dose if BG above target
        correction_dose = 0.0
        if current_bg > self.target_bg:
            correction_dose = (current_bg - self.target_bg) / self.isf
        
        # Protein/fat insulin impact (simple approximation)
        protein_fat_impact = ((total_protein * 0.1) + (total_fat * 0.2)) / self.icr
        
        # Sum up
        base_dose = carb_dose + correction_dose + protein_fat_impact
        
        # Time-of-day factor
        time_factors = {
            'breakfast': 1.1,  # Dawn phenomenon
            'lunch': 1.0,
            'dinner': 0.9
        }
        factor = time_factors.get(meal_time, 1.0)
        final_dose = base_dose * factor
        
        # Round to nearest 0.5
        final_dose = round(final_dose * 2) / 2
        
        # Generate BG prediction
        prediction_df = self._generate_glucose_prediction(
            start_bg=current_bg,
            insulin_dose=final_dose,
            carbs=total_carbs,
            gi=weighted_gi,
            protein=total_protein,
            fat=total_fat
        )
        
        details = {
            'total_carbs': round(total_carbs, 1),
            'weighted_gi': round(weighted_gi, 1),
            'carb_dose': round(carb_dose, 1),
            'correction_dose': round(correction_dose, 1),
            'protein_fat_impact': round(protein_fat_impact, 1),
            'final_dose': final_dose
        }
        
        return final_dose, details, prediction_df

    def _generate_glucose_prediction(self,
                                     start_bg: float,
                                     insulin_dose: float,
                                     carbs: float,
                                     gi: float,
                                     protein: float,
                                     fat: float) -> pd.DataFrame:
        """
        Simple 6-hour BG prediction (5-min increments).
        - Gaussian curve for insulin action
        - Gaussian curve for carbs
        - Gaussian curve for protein/fat
        """
        n_points = 73  # 6 hours + 1
        times = pd.date_range(start=datetime.now(), periods=n_points, freq='5T')
        
        insulin_effect = np.zeros(n_points)
        carb_effect = np.zeros(n_points)
        protein_fat_effect = np.zeros(n_points)
        
        # Insulin action peak ~1.5h => index ~18
        peak_insulin_idx = int(1.5 * 60 / 5)
        insulin_sd = 4
        for i in range(n_points):
            insulin_effect[i] = (norm.pdf(i, peak_insulin_idx, insulin_sd)
                                 * insulin_dose * self.isf * 5)
        
        # Carbs: peak ~1.5h if GI>70, else 2h
        if gi > 70:
            peak_carb_hours = 1.5
            carb_sd = 5
        else:
            peak_carb_hours = 2.0
            carb_sd = 7
        
        peak_carb_idx = int(peak_carb_hours * 60 / 5)
        for i in range(n_points):
            carb_effect[i] = (norm.pdf(i, peak_carb_idx, carb_sd)
                              * carbs * 4)  # 1g => ~4 mg/dL (approx)
        
        # Protein/fat: peak ~4h
        peak_pf_idx = int(4.0 * 60 / 5)
        pf_sd = 8
        pf_factor = (protein + fat) * 2
        for i in range(n_points):
            protein_fat_effect[i] = norm.pdf(i, peak_pf_idx, pf_sd) * pf_factor
        
        # Compute predicted BG
        predicted_bg = []
        current_val = start_bg
        for i in range(n_points):
            delta = carb_effect[i] + protein_fat_effect[i] - insulin_effect[i]
            current_val += delta
            predicted_bg.append(current_val)
        
        df = pd.DataFrame({
            'time': times,
            'predicted_bg': predicted_bg,
            'insulin_impact': -np.cumsum(insulin_effect),
            'carb_impact': np.cumsum(carb_effect),
            'protein_fat_impact': np.cumsum(protein_fat_effect),
        })
        return df

# -------------
# Plot Function
# -------------
def plot_prediction(prediction_df: pd.DataFrame):
    """Plot the predicted glucose curve + sub-components."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        prediction_df['time'],
        prediction_df['predicted_bg'],
        'k-',
        label='Predicted BG',
        lw=2
    )
    ax.plot(
        prediction_df['time'],
        prediction_df['insulin_impact'],
        'r--',
        label='Cumulative Insulin Effect',
        alpha=0.7
    )
    ax.plot(
        prediction_df['time'],
        prediction_df['carb_impact'],
        'g--',
        label='Cumulative Carb Effect',
        alpha=0.7
    )
    ax.plot(
        prediction_df['time'],
        prediction_df['protein_fat_impact'],
        'b--',
        label='Protein/Fat Effect',
        alpha=0.7
    )
    
    # Target range shading
    ax.axhspan(70, 180, color='green', alpha=0.1, label='Target BG Range')
    
    ax.set_title("Predicted 6-hour Glucose Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Blood Glucose (mg/dL)")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

# -------------
# Streamlit App
# -------------
def main():
    # Sidebar
    st.sidebar.title("Insulin Calculator Settings")
    st.sidebar.markdown(DISCLAIMER)
    st.sidebar.write("Use responsibly under medical guidance.")
    
    # Page title
    st.title("Enhanced Insulin Calculator for Ms. Padmaja Komaragiri")
    st.subheader("With Simplified PK/PD + South Indian Meal Carbs")
    
    # Initialize the calculator
    calc = InsulinCalculator(patient_weight=92.0)
    
    # Display baseline info
    with st.expander("Show Patient & Medication Details"):
        st.write(f"**Tresiba Dose:** {calc.current_tresiba} units")
        st.write(f"**ICR (Insulin-to-Carb Ratio):** {calc.icr} g carbs/unit")
        st.write(f"**ISF (Insulin Sensitivity Factor):** {calc.isf} mg/dL per unit")
        st.write("**Medications**: Janumet + Jardiance => synergy")
        st.write(f"**Target BG:** {calc.target_bg} mg/dL")
    
    # Current BG
    current_bg = st.slider(
        "Current Blood Glucose (mg/dL)",
        min_value=60,
        max_value=400,
        value=180,
        step=5
    )
    
    # Meal time
    meal_time = st.selectbox("Select Meal Time", ["breakfast", "lunch", "dinner"])
    
    st.write("---")
    st.write("### Enter Meal Contents")
    st.write("Specify how many cups/pieces of each item you plan to eat:")
    
    # Generate dynamic inputs
    all_foods = list(calc.food_db.foods.keys())
    meal_contents = {}
    for food in all_foods:
        fd = calc.food_db.foods[food]
        if 'carbs_per_cup' in fd:
            unit = "cups"
        else:
            unit = "pieces"
        qty = st.number_input(
            f"{food} ({unit})",
            min_value=0.0,
            value=0.0,
            step=0.5,
            key=food
        )
        if qty > 0:
            meal_contents[food] = qty
    
    # Calculate when button is clicked
    if st.button("Calculate & Predict Curve"):
        dose, details, prediction_df = calc.calculate_meal_insulin(
            current_bg=current_bg,
            meal_contents=meal_contents,
            meal_time=meal_time
        )
        st.success(f"**Recommended NovoRapid Dose: {dose} units**")
        st.write("#### Calculation Breakdown")
        st.write(details)
        
        # Plot
        st.write("### Predicted 6-hour Glucose Curve")
        plot_prediction(prediction_df)

    st.markdown("---")
    st.info("**Note:** Demo purposes only. Verify all doses with your doctor.")

if __name__ == "__main__":
    main()