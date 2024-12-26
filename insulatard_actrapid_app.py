##############################################
# insulatard_actrapid_app.py
##############################################

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
from typing import Dict, Tuple, List
from scipy.stats import norm
import seaborn as sns

# ------------------------------------------------
# Medical Disclaimer
# ------------------------------------------------
DISCLAIMER = """
**Medical Disclaimer**:
- This application is for **educational/demo** purposes only.
- It should **NOT** be used as a substitute for professional medical advice.
- Always consult your healthcare provider before making changes to your insulin regimen.
"""


# ------------------------------------------------
# 1. Pharmacology Class (Insulatard + Actrapid)
# ------------------------------------------------
class InsulinPharmacology:
    """Pharmacokinetic/Pharmacodynamic parameters for insulin types, plus oral medication synergy."""
    
    def __init__(self):
        # Actrapid (Regular/Soluble insulin) PK/PD
        self.actrapid_params = {
            'onset_minutes': 30,
            'peak_hours': 2.5,        # Typically peaks ~2-4h
            'duration_hours': 6,      # 5-8h total duration
            'half_life_minutes': 60,
            # Simplified action profile for interpolation
            'action_profile': {
                '0h': 0.0,
                '1h': 0.2,
                '2h': 0.4,
                '3h': 1.0,  # peak around 3h
                '4h': 0.8,
                '5h': 0.4,
                '6h': 0.1,
                '7h': 0.0
            }
        }
        
        # Insulatard (NPH/Isophane insulin) PK/PD
        self.insulatard_params = {
            'onset_hours': 1.5,
            'peak_hours': 4.5,        # Typically peaks ~4-6h
            'duration_hours': 16,     # 14-18h total duration
            # Simplified action profile for interpolation
            'action_profile': {
                '0h': 0.0,
                '2h': 0.2,
                '4h': 0.6,
                '6h': 1.0,  # peak
                '8h': 0.8,
                '12h': 0.4,
                '16h': 0.1,
                '18h': 0.0
            },
            'overlap_factor': 1.2  # if doses overlap, e.g. morning + evening
        }
        
        # Janumet (Sitagliptin + Metformin) synergy
        self.janumet_params = {
            'sitagliptin_half_life_hours': 12.4,
            'metformin_half_life_hours': 6.2,
            'insulin_sensitivity_increase': 0.15,  # ~15% improvement in sensitivity
            'peak_effect_hours': 4
        }
        
        # Jardiance (Empagliflozin) synergy
        self.jardiance_params = {
            'glucose_excretion_rate': 80,        # mg/dL per day excreted
            'insulin_sensitivity_increase': 0.10 # ~10% improvement in sensitivity
        }


# ------------------------------------------------
# 2. Regimen Class (Doses + Overlap Calculations)
# ------------------------------------------------
class InsulinRegimen:
    """
    Manages insulin dosing schedule and calculates overlapping action
    of Insulatard (NPH) and Actrapid (Regular).
    """
    def __init__(self):
        self.pharmacology = InsulinPharmacology()
        
        # Default doses (can be changed by user)
        self.morning_insulatard = 8.0
        self.evening_insulatard = 8.0
        self.breakfast_actrapid = 6.0
        self.lunch_actrapid = 6.0
        self.dinner_actrapid = 6.0
        
        # Typical injection schedule (24h clock)
        self.schedule = {
            'morning_insulatard': time(7, 0),   # 07:00
            'evening_insulatard': time(21, 0),  # 21:00
            'breakfast_actrapid': time(7, 30),  # 07:30
            'lunch_actrapid': time(13, 0),      # 13:00
            'dinner_actrapid': time(19, 0)      # 19:00
        }
    
    def total_insulin_action(self, t: datetime) -> float:
        """
        Calculate total insulin 'action' (relative effect) at a specific time 't'
        from all doses of Insulatard + Actrapid. We sum up the interpolated
        action profiles for each injection if they are within range.
        """
        total_action = 0.0
        
        # Evaluate Insulatard (morning + evening)
        total_action += self._insulatard_action(
            dose_amount=self.morning_insulatard,
            dose_time=self.schedule['morning_insulatard'],
            now=t
        )
        total_action += self._insulatard_action(
            dose_amount=self.evening_insulatard,
            dose_time=self.schedule['evening_insulatard'],
            now=t
        )
        
        # Evaluate Actrapid (breakfast + lunch + dinner)
        total_action += self._actrapid_action(
            dose_amount=self.breakfast_actrapid,
            dose_time=self.schedule['breakfast_actrapid'],
            now=t
        )
        total_action += self._actrapid_action(
            dose_amount=self.lunch_actrapid,
            dose_time=self.schedule['lunch_actrapid'],
            now=t
        )
        total_action += self._actrapid_action(
            dose_amount=self.dinner_actrapid,
            dose_time=self.schedule['dinner_actrapid'],
            now=t
        )
        
        return total_action
    
    # ---------------------------------
    # Helper: Insulatard action
    # ---------------------------------
    def _insulatard_action(self, dose_amount: float, dose_time: time, now: datetime) -> float:
        """Interpolate Insulatard effect for a given dose at time 'dose_time'."""
        if dose_amount <= 0:
            return 0.0
        
        # Calculate hours since injection
        injection_dt = now.replace(hour=dose_time.hour, minute=dose_time.minute, second=0)
        if injection_dt > now:
            # If injection time is "later" in the day than current time, subtract one day
            injection_dt = injection_dt - timedelta(days=1)
        
        hours_since = (now - injection_dt).total_seconds() / 3600.0
        if hours_since < 0 or hours_since > 18:
            # Outside of NPH action window
            return 0.0
        
        # Interpolate from the action_profile
        profile = self.pharmacology.insulatard_params['action_profile']
        hrs = np.array([float(h[:-1]) for h in profile.keys()])  # parse "2h" -> 2.0
        vals = np.array(list(profile.values()), dtype=float)
        effect_factor = np.interp(hours_since, hrs, vals)
        return dose_amount * effect_factor
    
    # ---------------------------------
    # Helper: Actrapid action
    # ---------------------------------
    def _actrapid_action(self, dose_amount: float, dose_time: time, now: datetime) -> float:
        """Interpolate Actrapid effect for a given dose at time 'dose_time'."""
        if dose_amount <= 0:
            return 0.0
        
        # Calculate hours since injection
        injection_dt = now.replace(hour=dose_time.hour, minute=dose_time.minute, second=0)
        if injection_dt > now:
            # If the injection was 'in the future' (same day), subtract 1 day
            injection_dt -= timedelta(days=1)
        
        hours_since = (now - injection_dt).total_seconds() / 3600.0
        if hours_since < 0 or hours_since > 7:
            # Outside typical Actrapid window
            return 0.0
        
        # Interpolate from the action_profile
        profile = self.pharmacology.actrapid_params['action_profile']
        hrs = np.array([float(h[:-1]) for h in profile.keys()])
        vals = np.array(list(profile.values()), dtype=float)
        effect_factor = np.interp(hours_since, hrs, vals)
        return dose_amount * effect_factor


# ------------------------------------------------
# 3. South Indian Food Database
# ------------------------------------------------
class SouthIndianFoodDatabase:
    """Database of South Indian foods with carbs, GI, protein, fat, etc."""
    
    def __init__(self):
        self.foods = {
            'rice': {
                'carbs_per_cup': 45,
                'gi': 73,
                'protein_g': 4.2,
                'fat_g': 0.4,
                'fiber_g': 0.6,
                'absorption_rate': 'rapid'
            },
            'idli': {
                'carbs_per_piece': 12,
                'gi': 68,
                'protein_g': 2.0,
                'fat_g': 0.1,
                'fiber_g': 0.8,
                'absorption_rate': 'medium'
            },
            'dosa': {
                'carbs_per_piece': 30,
                'gi': 69,
                'protein_g': 3.5,
                'fat_g': 1.5,
                'fiber_g': 1.2,
                'absorption_rate': 'medium-rapid'
            },
            'sambar': {
                'carbs_per_cup': 15,
                'gi': 45,
                'protein_g': 3.0,
                'fat_g': 2.0,
                'fiber_g': 4.5,
                'absorption_rate': 'slow'
            },
            'upma': {
                'carbs_per_cup': 35,
                'gi': 65,
                'protein_g': 3.5,
                'fat_g': 4.0,
                'fiber_g': 2.0,
                'absorption_rate': 'medium'
            },
            'vada': {
                'carbs_per_piece': 18,
                'gi': 72,
                'protein_g': 4.5,
                'fat_g': 9.0,
                'fiber_g': 1.5,
                'absorption_rate': 'rapid'
            },
            'coffee_with_milk': {
                'carbs_per_cup': 12,
                'gi': 35,
                'protein_g': 3.3,
                'fat_g': 3.2,
                'fiber_g': 0,
                'absorption_rate': 'rapid'
            },
            'ice_cream': {
                'carbs_per_cup': 32,
                'gi': 61,
                'protein_g': 3.5,
                'fat_g': 11.0,
                'fiber_g': 0.7,
                'absorption_rate': 'medium-rapid'
            }
        }


# ------------------------------------------------
# 4. Master Calculator (Mealtime, BG predictions)
# ------------------------------------------------
class InsulinCalculator:
    """
    Main calculator to:
      - Compute mealtime insulin based on carbs, ICR, correction factor
      - Predict a simplified BG curve over 6 hours
      - Visualize overlapping Insulin action (Actrapid + Insulatard)
    """
    def __init__(self, 
                 regimen: InsulinRegimen,
                 food_db: SouthIndianFoodDatabase,
                 total_daily_dose: float = 60.0,  # Example
                 target_bg: float = 140.0,
                 current_bg: float = 180.0):
        
        self.regimen = regimen
        self.food_db = food_db
        self.TDD = total_daily_dose
        self.target_bg = target_bg
        self.current_bg = current_bg
        
        # Simplistic "ICR" (500 rule) + synergy from Janumet (15%) & Jardiance (10%)
        # => multiply TDD by (1 - 0.15 - 0.10) is a possible approach if synergy is big
        # For demonstration, let's just incorporate it into ICR & ISF:
        synergy_factor = (1 + 0.15 + 0.10)  # ~25% more insulin sensitive
        self.icr = round((500 / self.TDD) / synergy_factor, 1)  # carbs covered by 1u
        self.isf = round((1800 / self.TDD) / synergy_factor, 1) # mg/dL drop per 1u insulin

    # ---------------------------------
    # Calculate mealtime insulin dose
    # ---------------------------------
    def calculate_meal_dose(self,
                            meal_contents: Dict[str, float]) -> Tuple[float, Dict]:
        """
        1. Sum up total carbs from meal_contents
        2. Carb dose = total_carbs / ICR
        3. Correction dose = max(0, (current_bg - target_bg)/ISF)
        4. Add them up => round to nearest 0.5
        """
        total_carbs = 0.0
        for food, qty in meal_contents.items():
            fd = self.food_db.foods.get(food, None)
            if not fd:
                continue
            # If 'carbs_per_cup' in fd, else assume 'carbs_per_piece'
            if 'carbs_per_cup' in fd:
                carbs = fd['carbs_per_cup'] * qty
            else:
                carbs = fd['carbs_per_piece'] * qty
            total_carbs += carbs
        
        # Carb coverage
        carb_dose = total_carbs / self.icr
        
        # Correction
        correction_needed = max(0, self.current_bg - self.target_bg)
        correction_dose = correction_needed / self.isf
        
        raw_dose = carb_dose + correction_dose
        final_dose = round(raw_dose * 2) / 2  # nearest 0.5
        
        breakdown = {
            'total_carbs': round(total_carbs, 1),
            'carb_dose': round(carb_dose, 1),
            'correction_dose': round(correction_dose, 1),
            'final_dose': final_dose
        }
        return final_dose, breakdown
    
    # ---------------------------------
    # Predict 6-hour BG Curve
    # ---------------------------------
    def predict_bg_curve(self, meal_insulin_units: float, meal_carbs: float) -> pd.DataFrame:
        """
        Very simplistic 6-hour BG prediction:
         - Adds meal carb effect (peak ~ 1.5-2h if GI>70 else 2.5-3h)
         - Subtracts the mealtime Actrapid dose
         - Also adds baseline from Insulatard if relevant
        """
        n_points = 73  # 6 hours + 1, each 5 min
        start = datetime.now()
        times = [start + timedelta(minutes=5 * i) for i in range(n_points)]
        
        # We assume mealtime Actrapid is injected "now"
        # We'll do a quick 'action profile' for that mealtime Actrapid
        # For carbs, we do a normal distribution or a simple model
        # This is an approximation for demonstration.
        
        # Guesstimate GI-based absorption peak
        if meal_carbs < 1:
            meal_carbs = 0
        if meal_carbs > 0:
            # assume GI= ~70 => peak 1.5h, or 2h. We'll pick 2h for demonstration
            peak_carb_idx = int(2.0 * 60 / 5)  # 24 intervals
            carb_sd = 5
        else:
            peak_carb_idx = 999  # no carbs
            carb_sd = 1
        
        # Action profile for that mealtime Actrapid
        # We'll mimic the interpolation from the regimen class
        profile = self.regimen.pharmacology.actrapid_params['action_profile']
        actrapid_hrs = np.array([float(h[:-1]) for h in profile.keys()])
        actrapid_vals = np.array(list(profile.values()), dtype=float)
        
        # Accumulate arrays
        insulin_effect = np.zeros(n_points)
        carb_effect = np.zeros(n_points)
        predicted_bg = []
        
        # Start BG
        current_bg = float(self.current_bg)
        
        for i in range(n_points):
            # minutes -> hours
            hours_since_injection = i * 5 / 60.0
            if hours_since_injection <= 7:
                effect_factor = np.interp(hours_since_injection, actrapid_hrs, actrapid_vals)
                insulin_effect[i] = meal_insulin_units * self.isf * effect_factor * 5
            
            # carbs
            x = i  # index
            # Normal distribution approach
            carb_effect[i] = (norm.pdf(x, peak_carb_idx, carb_sd) * meal_carbs * 4)
            
            # Summation
            delta = carb_effect[i] - insulin_effect[i]
            current_bg += delta
            predicted_bg.append(current_bg)
        
        df = pd.DataFrame({
            'time': times,
            'predicted_bg': predicted_bg,
            'carb_impact': np.cumsum(carb_effect),
            'insulin_impact': -np.cumsum(insulin_effect)
        })
        return df


# ------------------------------------------------
# 5. Utility Plot Functions
# ------------------------------------------------
def plot_insulin_profiles(regimen: InsulinRegimen):
    """
    Plot 24 hours of overlapping Insulatard + Actrapid action, 
    given the user-specified doses and schedule.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Evaluate from now (0h) to next 24 hours in 30-min increments
    start = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [start + timedelta(minutes=30*i) for i in range(48)]  # 0..24h
    
    actions = []
    for t in times:
        val = regimen.total_insulin_action(t)
        actions.append(val)
    
    # Convert times to x-axis hours from now
    x_hours = [(t - start).total_seconds()/3600 for t in times]
    
    ax.plot(x_hours, actions, label='Total Insulin Action', color='blue', linewidth=2)
    ax.set_title("24-hour Overlapping Insulin Profile (Insulatard + Actrapid)")
    ax.set_xlabel("Hours from now")
    ax.set_ylabel("Relative Insulin Action")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)


def plot_bg_prediction(df: pd.DataFrame):
    """
    Plot the predicted 6-hour BG curve from the mealtime dose.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df['time'], df['predicted_bg'], 'k-', label='Predicted BG', lw=2)
    
    # Show cumulative carb vs. insulin impact
    ax.plot(df['time'], df['carb_impact'], 'g--', label='Cumulative Carb Impact')
    ax.plot(df['time'], df['insulin_impact'], 'r--', label='Cumulative Insulin Impact')
    
    # Target range shading
    ax.axhspan(70, 180, color='green', alpha=0.1, label='Target Range')
    
    ax.set_title("6-hour Predicted Glucose Curve (Mealtime Actrapid)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Blood Glucose (mg/dL)")
    plt.xticks(rotation=30)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


# ------------------------------------------------
# 6. Streamlit App
# ------------------------------------------------
def main():
    st.sidebar.title("Insulin Calculator Settings")
    st.sidebar.markdown(DISCLAIMER)
    st.sidebar.info("For demonstration only. Consult your doctor for real dosing!")
    
    st.title("Insulatard & Actrapid Insulin Calculator")
    st.subheader("With South Indian Meals & Simplified PK/PD")
    
    # Initialize
    regimen = InsulinRegimen()
    food_db = SouthIndianFoodDatabase()
    
    # Let user modify default doses
    st.markdown("## Insulin Doses")
    regimen.morning_insulatard = st.number_input("Morning Insulatard (units)", 0.0, 100.0, 8.0, 1.0)
    regimen.evening_insulatard = st.number_input("Evening Insulatard (units)", 0.0, 100.0, 8.0, 1.0)
    regimen.breakfast_actrapid = st.number_input("Breakfast Actrapid (units)", 0.0, 50.0, 6.0, 1.0)
    regimen.lunch_actrapid = st.number_input("Lunch Actrapid (units)", 0.0, 50.0, 6.0, 1.0)
    regimen.dinner_actrapid = st.number_input("Dinner Actrapid (units)", 0.0, 50.0, 6.0, 1.0)
    
    if st.button("Plot 24h Overlapping Insulin Action"):
        plot_insulin_profiles(regimen)
    
    st.markdown("---")
    st.markdown("## Mealtime Insulin Calculation")
    
    total_daily_dose = st.number_input("Approx. Total Daily Insulin (units)", 10.0, 200.0, 60.0, 1.0)
    current_bg = st.slider("Current Blood Glucose (mg/dL)", 50, 400, 180, 5)
    target_bg = st.number_input("Target Blood Glucose (mg/dL)", 70, 300, 140, 5)
    
    # Prepare main calculator
    calc = InsulinCalculator(regimen, food_db, total_daily_dose, target_bg, current_bg)
    
    st.markdown("### Enter Meal Contents")
    meal_options = list(food_db.foods.keys())
    meal_contents = {}
    for food in meal_options:
        fd = food_db.foods[food]
        # Decide if "cup" or "piece"
        if "carbs_per_cup" in fd:
            unit = "cups"
        else:
            unit = "pieces"
        qty = st.number_input(f"{food} ({unit})", min_value=0.0, value=0.0, step=0.5)
        if qty > 0:
            meal_contents[food] = qty
    
    if st.button("Calculate Mealtime Dose + Predict BG"):
        dose, details = calc.calculate_meal_dose(meal_contents)
        
        st.success(f"**Recommended Mealtime Actrapid Dose**: {dose} units")
        st.write("**Calculation Details:**", details)
        
        # For BG prediction, let's assume the user is about to take that mealtime Actrapid
        # and eat the meal's carbs
        df_pred = calc.predict_bg_curve(meal_insulin_units=dose, 
                                        meal_carbs=details['total_carbs'])
        plot_bg_prediction(df_pred)
    
    st.info("**Disclaimer**: This is a simplified model; actual insulin absorption and BG responses vary.")

if __name__ == "__main__":
    main()