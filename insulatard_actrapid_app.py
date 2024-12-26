##############################################
# insulatard_actrapid_app.py
##############################################
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time, date
from typing import Dict, Tuple
from scipy.stats import norm
import seaborn as sns

##############################
# 0. Medical Disclaimer
##############################
DISCLAIMER = """
<div style="background-color: #ffecec; padding: 15px; border-radius: 5px;">
<h3 style="color: #d9534f; margin-bottom: 8px;">Medical Disclaimer</h3>
<p style="color: #333; line-height:1.4;">
This application is for <b>educational/demo</b> purposes only. 
It should <b>NOT</b> be used as a substitute for professional medical advice 
or emergency treatment. Always consult your healthcare provider or endocrinologist
before making any changes to your insulin or medication regimen.<br><br>
Due to intra- and inter-individual pharmacokinetic variability, real clinical results
can differ from these theoretical models. 
Data references include:
<ul>
<li>Hirsch IB. Glycemic Variability and Diabetes Complications. <i>Diabetes Technol Ther.</i> 2020.</li>
<li>Heinemann L. Insulin absorption and therapy. <i>Diabetes Obes Metab.</i> 2015.</li>
<li>American Diabetes Association. <i>Standards of Medical Care in Diabetes—2023.</i></li>
</ul>
</p>
</div>
"""

##############################
# 0.1 Custom CSS for UI
##############################
CUSTOM_CSS = """
<style>
body {
    background-color: #f7f9fc;
    font-family: "Source Sans Pro", sans-serif;
    color: #2c3e50;
}
.sidebar .sidebar-content {
    background: #ffffff;
    color: #2c3e50;
}
footer {
    visibility: hidden;
}
h1, h2, h3, h4, h5, h6 {
    font-family: "Source Sans Pro", sans-serif;
    color: #2c3e50;
}
hr {
    border: 1px solid #ddd;
}
.stNumberInput > label, .stSlider > label {
    font-weight: 600;
}
div.stButton > button:first-child {
    background-color: #1e88e5;
    color: white;
    padding: 0.7rem 1rem;
    border-radius: 5px;
    border: none;
    font-size: 1rem;
}
div.stButton > button:hover {
    background-color: #1565c0;
    color: white;
}
</style>
"""

########################################################################
# 1. Pharmacology Classes & PK Curves
########################################################################

class InsulinPharmacology:
    """
    Detailed pharmacokinetic parameters for:
    - Actrapid (Regular insulin)
    - Insulatard (NPH insulin)
    Using piecewise or gamma-like approximations, referencing typical research data.
    """

    def __init__(self):
        # Advanced piecewise approximation for Actrapid (Regular):
        self.actrapid_curve = {
            0.0: 0.0,
            0.5: 0.05,
            1.0: 0.35,
            1.5: 0.70,
            2.0: 1.0,   # peak ~2h
            2.5: 0.95,
            3.0: 0.80,
            4.0: 0.45,
            5.0: 0.25,
            6.0: 0.10,
            7.0: 0.0
        }
        
        # Piecewise approximation for Insulatard (NPH):
        self.insulatard_curve = {
            0.0: 0.0,
            1.0: 0.0,
            2.0: 0.2,
            4.0: 0.7,
            6.0: 1.0,   # peak ~4-8h
            8.0: 0.85,
            10.0: 0.70,
            12.0: 0.50,
            14.0: 0.30,
            16.0: 0.15,
            18.0: 0.05,
            20.0: 0.0
        }

    def get_actrapid_effect_factor(self, hours_since_injection: float) -> float:
        """
        Returns an interpolated effect factor (0..1) for Actrapid
        at a given time post-injection.
        """
        if hours_since_injection < 0 or hours_since_injection > 7:
            return 0.0
        x_pts = np.array(list(self.actrapid_curve.keys()))
        y_pts = np.array(list(self.actrapid_curve.values()))
        return np.interp(hours_since_injection, x_pts, y_pts)

    def get_insulatard_effect_factor(self, hours_since_injection: float) -> float:
        """
        Returns an interpolated effect factor (0..1) for Insulatard (NPH)
        at a given time post-injection.
        """
        if hours_since_injection < 0 or hours_since_injection > 20:
            return 0.0
        x_pts = np.array(list(self.insulatard_curve.keys()))
        y_pts = np.array(list(self.insulatard_curve.values()))
        return np.interp(hours_since_injection, x_pts, y_pts)


########################################################################
# 2. Insulin Regimen & Overlapping Action
########################################################################

class InsulinRegimen:
    """
    Manages:
    - Doses of Insulatard (morning/evening)
    - Doses of Actrapid (breakfast/lunch/dinner)
    - Injection times
    - Summation of the insulin action from each injection.
    """

    def __init__(self):
        self.pharmacology = InsulinPharmacology()
        
        # Default doses (units)
        self.morning_insulatard = 8.0
        self.evening_insulatard = 8.0
        
        self.breakfast_actrapid = 6.0
        self.lunch_actrapid = 6.0
        self.dinner_actrapid = 6.0
        
        # Default injection times
        self.schedule = {
            'morning_insulatard': time(7, 0),
            'evening_insulatard': time(21, 0),
            'breakfast_actrapid': time(7, 30),
            'lunch_actrapid': time(13, 0),
            'dinner_actrapid': time(19, 0)
        }

    def total_insulin_action(self, t: datetime) -> float:
        """
        Summation of insulin action from all scheduled injections at time 't'.
        """
        total_action = 0.0
        
        total_action += self._nph_action(
            dose_amount=self.morning_insulatard,
            dose_time=self.schedule['morning_insulatard'],
            now=t
        )
        total_action += self._nph_action(
            dose_amount=self.evening_insulatard,
            dose_time=self.schedule['evening_insulatard'],
            now=t
        )
        
        total_action += self._regular_action(
            dose_amount=self.breakfast_actrapid,
            dose_time=self.schedule['breakfast_actrapid'],
            now=t
        )
        total_action += self._regular_action(
            dose_amount=self.lunch_actrapid,
            dose_time=self.schedule['lunch_actrapid'],
            now=t
        )
        total_action += self._regular_action(
            dose_amount=self.dinner_actrapid,
            dose_time=self.schedule['dinner_actrapid'],
            now=t
        )
        
        return total_action

    #######################
    # Helper Methods
    #######################
    def _nph_action(self, dose_amount: float, dose_time: time, now: datetime) -> float:
        if dose_amount <= 0:
            return 0.0
        injection_dt = now.replace(hour=dose_time.hour, minute=dose_time.minute, second=0, microsecond=0)
        if injection_dt > now:
            # If the chosen time is "later" than now, interpret as yesterday
            injection_dt -= timedelta(days=1)
        hours_since = (now - injection_dt).total_seconds() / 3600.0
        effect_factor = self.pharmacology.get_insulatard_effect_factor(hours_since)
        return dose_amount * effect_factor

    def _regular_action(self, dose_amount: float, dose_time: time, now: datetime) -> float:
        if dose_amount <= 0:
            return 0.0
        injection_dt = now.replace(hour=dose_time.hour, minute=dose_time.minute, second=0, microsecond=0)
        if injection_dt > now:
            # If the chosen time is "later" than now, interpret as yesterday
            injection_dt -= timedelta(days=1)
        hours_since = (now - injection_dt).total_seconds() / 3600.0
        effect_factor = self.pharmacology.get_actrapid_effect_factor(hours_since)
        return dose_amount * effect_factor


########################################################################
# 3. Food Database
########################################################################

class DetailedFoodDatabase:
    """
    Contains the detailed breakdown from your provided macronutrient list.
    """

    def __init__(self):
        self.foods = {
            # Breakfast items
            "idli": {
                "carbs": 8.3,   # grams per piece
                "protein": 1.3,
                "fat": 0.17,
                "calories": 40
            },
            "coffee_cup": {
                "carbs": 2,     # grams per 1 cup
                "protein": 1,
                "fat": 0.5,
                "calories": 20
            },
            "sambar_cup": {
                "carbs": 12,
                "protein": 4,
                "fat": 3,
                "calories": 100
            },
            "ensure_dm_scoop": {
                "carbs": 13,
                "protein": 4,
                "fat": 3,
                "calories": 100
            },
            "mixed_juice_glass": {
                "carbs": 16.5,
                "protein": 1,
                "fat": 0,
                "calories": 70
            },
            # Lunch/Dinner items
            "rice_cup": {
                "carbs": 57.5,
                "protein": 5,
                "fat": 0.5,
                "calories": 260
            },
            "dal_cup": {
                "carbs": 27,
                "protein": 12,
                "fat": 3,
                "calories": 180
            },
            "veg_curry_cup": {
                "carbs": 12,
                "protein": 2,
                "fat": 4,
                "calories": 100
            }
        }


########################################################################
# 4. Insulin Calculator (Meal Bolus, Correction, BG Prediction)
########################################################################

class InsulinCalculator:
    """
    For each meal:
     1) Summation of total carbs
     2) Mealtime insulin dose = (total carbs / I:C ratio) + correction
     3) 6-hour BG prediction, factoring in injection time
    """

    def __init__(
        self,
        regimen: InsulinRegimen,
        food_db: DetailedFoodDatabase,
        total_daily_dose: float,
        current_bg: float,
        target_bg: float,
        addl_sensitivity_percent: float = 0.0
    ):
        self.regimen = regimen
        self.food_db = food_db
        self.TDD = total_daily_dose
        self.current_bg = current_bg
        self.target_bg = target_bg
        self.addl_sensitivity = addl_sensitivity_percent / 100.0  # e.g. 10% => 0.1

        # 500 rule => ICR, 1800 rule => ISF, scaled by additional sensitivity
        factor = 1.0 + self.addl_sensitivity
        effective_tdd = self.TDD / factor
        self.icr = 500.0 / effective_tdd  # grams carb per 1 unit
        self.isf = 1800.0 / effective_tdd # mg/dL drop per 1 unit

        # We clamp BG at a minimum ~40 mg/dL to avoid negative
        self.min_bg_clamp = 40.0

    def calculate_meal_dose(self, meal_contents: Dict[str, float]) -> Tuple[float, Dict]:
        """
        1. total_carbs = sum of carbs from selected items
        2. carb_dose = total_carbs / ICR
        3. correction_dose = max(0, (current_bg - target_bg) / ISF)
        4. round to nearest 0.5
        """
        total_carbs = 0.0
        for food, qty in meal_contents.items():
            if food not in self.food_db.foods:
                continue
            data = self.food_db.foods[food]
            total_carbs += data["carbs"] * qty

        carb_dose = total_carbs / self.icr if self.icr != 0 else 0.0
        correction_needed = max(0, self.current_bg - self.target_bg)
        correction_dose = correction_needed / self.isf if self.isf != 0 else 0.0
        
        raw_dose = carb_dose + correction_dose
        final_dose = round(raw_dose * 2) / 2

        breakdown = {
            "total_carbs_g": round(total_carbs, 1),
            "carb_dose_units": round(carb_dose, 1),
            "correction_dose_units": round(correction_dose, 1),
            "final_recommended_units": final_dose
        }
        return final_dose, breakdown

    def predict_bg_curve(
        self,
        meal_insulin_units: float,
        meal_carbs_g: float,
        injection_time: time,
        duration_hours: float = 6.0,
        plot_resolution_minutes: int = 5
    ) -> pd.DataFrame:
        """
        Predict BG changes over 'duration_hours' (default 6h),
        factoring the user's chosen injection time for this meal.
         - Normal PDF for carb absorption (peak ~2h)
         - Actrapid piecewise PK, offset by injection time
         - BG clamped at 40 mg/dL
        """
        # We'll assume the meal is eaten near the injection time.
        # Let's create a baseline "now" (the moment we run the calc),
        # then interpret the injection as today at 'injection_time'.
        now = datetime.now().replace(second=0, microsecond=0)
        injection_dt = now.replace(hour=injection_time.hour, minute=injection_time.minute)
        if injection_dt < now - timedelta(hours=12):
            # If the user sets a time that's far behind, assume it's from 'yesterday'
            injection_dt += timedelta(days=1)
        elif injection_dt > now + timedelta(hours=12):
            # If user sets a time in the far future, assume it's from 'yesterday' 
            injection_dt -= timedelta(days=1)

        n_points = int(duration_hours*60 / plot_resolution_minutes) + 1
        times = [now + timedelta(minutes=i*plot_resolution_minutes) for i in range(n_points)]

        predicted_bg = []
        current_bg = float(self.current_bg)

        insulin_effect_array = np.zeros(n_points)
        carb_effect_array = np.zeros(n_points)

        # peak carb ~2h, stdev ~0.7
        carb_peak = 2.0
        carb_sd = 0.7

        for i in range(n_points):
            t_now = times[i]
            # Hours since injection
            dt_hours = (t_now - injection_dt).total_seconds()/3600.0

            # Insulin effect factor
            actrapid_factor = self.regimen.pharmacology.get_actrapid_effect_factor(dt_hours)
            # mg/dL drop at this interval
            insulin_drop = meal_insulin_units * self.isf * actrapid_factor * (plot_resolution_minutes/60.0)
            insulin_effect_array[i] = insulin_drop

            # Carb absorption: we assume meal is eaten roughly at the same time as injection
            # so we anchor the "carb curve" at injection_dt as well.
            if dt_hours >= 0:
                # Only after injection time
                pdf_val = norm.pdf(dt_hours, loc=carb_peak, scale=carb_sd)
                total_bg_rise_if_no_insulin = meal_carbs_g * 4.0
                carb_increment = total_bg_rise_if_no_insulin * pdf_val * (plot_resolution_minutes/60.0)
                carb_effect_array[i] = carb_increment
            else:
                # before injection => no carb
                carb_effect_array[i] = 0.0

        for i in range(n_points):
            delta = carb_effect_array[i] - insulin_effect_array[i]
            current_bg += delta
            if current_bg < self.min_bg_clamp:
                current_bg = self.min_bg_clamp
            predicted_bg.append(current_bg)

        df = pd.DataFrame({
            "time": times,
            "predicted_bg_mgdl": predicted_bg,
            "carb_increment": carb_effect_array,
            "insulin_decrement": insulin_effect_array
        })
        return df


########################################################################
# 5. Utility Plot Functions
########################################################################

def plot_24h_insulin_action(regimen: InsulinRegimen):
    """Plot 24-hour overlapping insulin action from the user-defined schedule."""
    fig, ax = plt.subplots(figsize=(10, 5))
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    times_24h = [now + timedelta(minutes=30*i) for i in range(48)]  # 24h in 30-min steps

    total_actions = []
    x_hours = []

    for t in times_24h:
        hours_from_now = (t - now).total_seconds()/3600.0
        total_actions.append(regimen.total_insulin_action(t))
        x_hours.append(hours_from_now)

    ax.plot(x_hours, total_actions, color='blue', lw=2, label='Total Insulin Action')
    ax.set_xlabel("Hours from Now")
    ax.set_ylabel("Relative Insulin Action (arbitrary units)")
    ax.set_title("24-hour Overlapping Insulin Action (NPH + Regular)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def plot_bg_prediction(df: pd.DataFrame, duration_hours: float = 6.0):
    """Plot the BG prediction curve plus cumulative carb & insulin effects."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time"], df["predicted_bg_mgdl"], 'k-', lw=2, label="Predicted BG (mg/dL)")
    ax.set_ylabel("Blood Glucose (mg/dL)")

    # For cumulative effect lines, we do cumsum
    cumsum_carb = np.cumsum(df["carb_increment"])
    cumsum_insulin = np.cumsum(df["insulin_decrement"])

    # We'll create a twin y-axis
    ax2 = ax.twinx()
    ax2.plot(df["time"], cumsum_carb, 'g--', label="Cumulative Carb Effect")
    ax2.plot(df["time"], -cumsum_insulin, 'r--', label="Cumulative Insulin Effect")
    ax2.set_ylabel("Cumulative Effect (mg/dL)")

    ax.axhspan(80, 180, color='green', alpha=0.1, label='Target Range')

    ax.set_title(f"{duration_hours}-hour Post-Meal BG Prediction")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    st.pyplot(fig)


########################################################################
# 6. Basal Calculator for Insulatard
########################################################################

def calculate_suggested_insulatard(total_daily_dose: float, fraction_basal: float = 0.4):
    """
    Basic tool: suggests how many total basal units (Insulatard) 
    based on fraction of TDD, then splits it morning/evening.
    By default, ~40% of TDD is basal, 60% is bolus in many T2DM regimens.
    """
    suggested_basal = total_daily_dose * fraction_basal
    # e.g., half in AM, half in PM
    morning = round(suggested_basal / 2.0)
    evening = round(suggested_basal / 2.0)
    return morning, evening, suggested_basal


########################################################################
# 7. Streamlit App
########################################################################

def display_mealtime_calculation(regimen, food_db, meal_label: str):
    """
    Renders a separate section for mealtime calculation (Breakfast, Lunch, Dinner).
    Asks for:
     - TDD
     - Additional Sensitivity
     - Current BG
     - Target BG
     - Injection Time (for PK offset)
     - Meal items
    Then calculates a recommended Actrapid dose + 6-hour BG curve.
    """
    st.markdown(f"### {meal_label} Calculation")
    st.write(f"Use the controls below to set your TDD, Additional Sensitivity, Current BG, and Target BG for **{meal_label}**. Then select your meal items. Lastly, specify the approximate insulin injection time for {meal_label} to see accurate PK/PD offsets.")

    # TDD, Sensitivity, BG
    total_daily_dose = st.number_input(
        f"{meal_label} - Estimated TDD (units)",
        min_value=10.0, max_value=300.0, value=60.0, step=1.0,
        key=f"{meal_label}_tdd"
    )
    addl_sens = st.slider(
        f"{meal_label} - Additional Sensitivity (%)",
        min_value=0, max_value=50, value=0, step=1,
        key=f"{meal_label}_sens"
    )
    current_bg = st.number_input(
        f"{meal_label} - Current BG (mg/dL)",
        min_value=50, max_value=600, value=150, step=5,
        key=f"{meal_label}_bg"
    )
    target_bg = st.number_input(
        f"{meal_label} - Target BG (mg/dL)",
        min_value=70, max_value=300, value=140, step=5,
        key=f"{meal_label}_tbg"
    )

    # Injection Time
    st.markdown(f"**Select {meal_label} Actrapid Injection Time**:")
    meal_injection_time = st.time_input(
        f"{meal_label} Injection Time",
        value=time(12, 0),  # default noon for lunch, for example
        key=f"{meal_label}_injtime"
    )

    calc = InsulinCalculator(regimen, food_db, total_daily_dose, current_bg, target_bg, addl_sens)

    st.markdown(f"#### Select {meal_label} Meal Contents")
    st.write(f"How many cups/pieces of each item are you about to have for {meal_label}?")

    meal_contents = {}
    for food_key, data in food_db.foods.items():
        qty = st.number_input(
            f"{meal_label} - {food_key} (qty)",
            min_value=0.0, max_value=10.0, step=0.5, value=0.0,
            key=f"{meal_label}_{food_key}"
        )
        if qty > 0:
            meal_contents[food_key] = qty

    if st.button(f"Calculate {meal_label} Bolus + Predict BG"):
        st.markdown(f"**{meal_label} - Recommended Mealtime Actrapid Dose**")
        dose, breakdown = calc.calculate_meal_dose(meal_contents)

        # Large green box for final recommendation
        st.markdown(
            f"""
            <div style="background-color: #d4edda; border-left: 6px solid #28a745; padding: 15px; margin: 10px 0; border-radius: 5px;">
            <h4 style='margin:0; color:#155724;'>Recommended Actrapid Dose: {dose} units</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("**Detailed Calculation:**")
        st.json(breakdown)

        st.markdown(f"**6-hour Post-{meal_label} BG Projection**")
        df_pred = calc.predict_bg_curve(
            meal_insulin_units=dose,
            meal_carbs_g=breakdown['total_carbs_g'],
            injection_time=meal_injection_time
        )
        plot_bg_prediction(df_pred, duration_hours=6.0)

        st.info(f"""
        This {meal_label} calculation is independent of any prior meal 
        (we do not automatically factor "insulin on board" from breakfast). 
        If you took insulin earlier, real BG may differ from this projection. 
        Always confirm final dosing with your physician.
        """)


def main():
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    st.sidebar.title("Insulin Calculator Settings")
    st.sidebar.markdown(DISCLAIMER, unsafe_allow_html=True)
    
    st.title("Advanced Insulatard (NPH) & Actrapid (Regular) Insulin Calculator")
    st.subheader("For Detailed Meal Planning, PK/PD Analysis & Multiple Meals")

    # Initialize
    regimen = InsulinRegimen()
    food_db = DetailedFoodDatabase()
    
    st.markdown("<hr/>", unsafe_allow_html=True)

    ##############################
    # Section: Basal Insulatard Calculator
    ##############################
    st.markdown("## 1) Basal Insulatard Calculator (Optional)")
    st.write("""
    Enter your <b>Total Daily Dose (TDD)</b> below, choose how much % you want as 
    <b>basal (NPH)</b>, and get a rough suggestion. 
    You can then set your actual doses in the sliders below.
    """, unsafe_allow_html=True)

    tdd_for_basal_calc = st.number_input(
        "Estimated TDD (units)", 10.0, 300.0, 60.0, 1.0
    )
    basal_fraction = st.slider(
        "Fraction of TDD as Basal (%)", 10, 60, 40, 5
    )
    
    if st.button("Suggest Basal Dosing"):
        morning_nph, evening_nph, total_basal = calculate_suggested_insulatard(
            tdd_for_basal_calc,
            fraction_basal=basal_fraction/100.0
        )
        st.success(
            f"**Suggested Basal (NPH) ~{int(total_basal)} units/day** "
            f"(e.g., {morning_nph} units AM + {evening_nph} units PM)."
        )
        st.info("You can input these values manually in the sliders below, or adjust as prescribed.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    ##############################
    # Section: Set Basal/Bolus in the Regimen
    ##############################
    st.markdown("## 2) Set Your Basal & Bolus Insulin Doses + Injection Times")
    st.write("""
    <p style="line-height:1.6;">
    <b>Note:</b> These values define your typical daily insulin regimen (NPH + Regular). 
    They also generate the "24-hour Overlapping Insulin Action" graph below to visualize 
    total coverage. 
    <br/>
    <em>This does NOT automatically change the mealtime calculations in the next section.</em>
    </p>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        regimen.morning_insulatard = st.number_input("Morning Insulatard (units)", 0.0, 200.0, 8.0, 1.0)
        regimen.evening_insulatard = st.number_input("Evening Insulatard (units)", 0.0, 200.0, 8.0, 1.0)
        morning_hour = st.slider("Morning Insulatard Hour (24h)", 0, 23, 7)
        regimen.schedule['morning_insulatard'] = time(morning_hour, 0)
        
        evening_hour = st.slider("Evening Insulatard Hour (24h)", 0, 23, 21)
        regimen.schedule['evening_insulatard'] = time(evening_hour, 0)
    
    with c2:
        regimen.breakfast_actrapid = st.number_input("Breakfast Actrapid (units)", 0.0, 200.0, 6.0, 1.0)
        regimen.lunch_actrapid = st.number_input("Lunch Actrapid (units)", 0.0, 200.0, 6.0, 1.0)
        regimen.dinner_actrapid = st.number_input("Dinner Actrapid (units)", 0.0, 200.0, 6.0, 1.0)
        
        breakfast_hour = st.slider("Breakfast Actrapid Hour (24h)", 0, 23, 7)
        regimen.schedule['breakfast_actrapid'] = time(breakfast_hour, 30)
        lunch_hour = st.slider("Lunch Actrapid Hour (24h)", 0, 23, 13)
        regimen.schedule['lunch_actrapid'] = time(lunch_hour, 0)
        dinner_hour = st.slider("Dinner Actrapid Hour (24h)", 0, 23, 19)
        regimen.schedule['dinner_actrapid'] = time(dinner_hour, 0)

    st.markdown("""
    <p style="color:#444; font-size:1rem;">
    Click below to see a 24-hour chart of the overlapping insulin action 
    from these basal+bolus doses.
    </p>
    """, unsafe_allow_html=True)
    
    if st.button("Plot 24-hour Overlapping Insulin Action"):
        plot_24h_insulin_action(regimen)

    st.markdown("<hr/>", unsafe_allow_html=True)

    ##############################
    # Section: Multiple Mealtime Calculations
    ##############################
    st.markdown("## 3) Multiple Mealtime Bolus Calculations")
    st.write("""
    <p style="line-height:1.6;">
    Use the tabs below to separately calculate your Actrapid dose for 
    <b>Breakfast</b>, <b>Lunch</b>, or <b>Dinner</b>. 
    If you already took the recommended dose for breakfast, just move to the 
    Lunch tab and enter your <em>new</em> Current BG, Target BG, Injection Time, 
    and meal contents for lunch, etc.
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Breakfast", "Lunch", "Dinner"])

    with tab1:
        display_mealtime_calculation(regimen, food_db, "Breakfast")

    with tab2:
        display_mealtime_calculation(regimen, food_db, "Lunch")

    with tab3:
        display_mealtime_calculation(regimen, food_db, "Dinner")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <h4 style='color:#2c3e50;'>Important Notes</h4>
    <ul style='font-size:1rem; color:#444; line-height:1.6;'>
    <li><b>Basal (Insulatard) Dosing:</b> Typically given morning + evening. 
        Adjust per your doctor's instructions or the "Basal Calculator."</li>
    <li><b>Multiple Meals / Insulin On Board:</b> Each meal tab's calculation 
        is independent. If you've already dosed for a previous meal, actual BG 
        might differ from these theoretical curves.</li>
    <li><b>Frequent Monitoring:</b> Check your blood glucose (fingerstick or CGM) 
        before meals, at bedtime, and as recommended. Hypoglycemia risk can be 
        significant if dosing is not individually tailored.</li>
    <li><b>Clinical Judgment:</b> Always confirm final insulin doses with your 
        healthcare provider. This tool is <b>not</b> a substitute for professional 
        medical advice or emergency care.</li>
    </ul>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
