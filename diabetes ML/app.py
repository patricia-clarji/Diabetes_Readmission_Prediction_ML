try:
    from flask import Flask, render_template, request
    import pandas as pd
    import pickle
except ImportError as e:
    print(f"Import error: {e}")
    raise

# Load models
model_readmit = pickle.load(open("notebooks/clf.pkl", "rb"))
model_time    = pickle.load(open("notebooks/reg.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    readmitted_prediction = None
    time_prediction       = None

    if request.method == "POST":

        # ── Admission type ────────────────────────────────────────────────
        admission_type_val = request.form["admission_type_id"]
        admission_type_id  = {"emergency": 1, "elective": 3}.get(admission_type_val, 6)

        # ── Discharge disposition ─────────────────────────────────────────
        discharge_val          = request.form["discharge_group"]
        discharge_disposition_id = {"home": 1, "care": 3, "expired": 11}.get(discharge_val, 6)

        # ── Admission source ──────────────────────────────────────────────
        source_val         = request.form["admission_source_group"]
        admission_source_id = {"referral": 1, "emergency": 7, "transfer": 4}.get(source_val, 9)

        # ── Numeric features ──────────────────────────────────────────────
        num_lab_procedures = int(request.form["num_lab_procedures"])
        num_procedures     = int(request.form["num_procedures"])
        num_medications    = int(request.form["num_medications"])
        number_outpatient  = int(request.form["number_outpatient"])
        number_emergency   = int(request.form["number_emergency"])
        number_inpatient   = int(request.form["number_inpatient"])
        number_diagnoses   = int(request.form["number_diagnoses"])
        age_numeric        = int(request.form["age_numeric"])

        # ── Engineered features (must match notebook exactly) ─────────────
        total_visits     = number_inpatient + number_emergency + number_outpatient
        total_procedures = num_procedures + num_lab_procedures

        # ── Binary / ordinal categoricals ─────────────────────────────────
        change      = 1 if request.form["change"]      == "Ch"  else 0
        diabetesMed = 1 if request.form["diabetesMed"] == "Yes" else 0

        insulin_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
        insulin     = insulin_map.get(request.form["insulin"], 0)

        # ── Race OHE ──────────────────────────────────────────────────────
        # Reference category (drop_first): AfricanAmerican → all zeros
        race_val       = request.form["race"]
        race_Caucasian = 1 if race_val == "Caucasian" else 0
        race_Missing   = 1 if race_val == "Missing"   else 0
        race_Other     = 1 if race_val == "Other"     else 0
        # AfricanAmerican → race_Caucasian=0, race_Missing=0, race_Other=0

        # ── Gender OHE ────────────────────────────────────────────────────
        # Reference category (drop_first): Female → 0
        gender_Male = 1 if request.form["gender"] == "Male" else 0

        # ── A1Cresult OHE ─────────────────────────────────────────────────
        # Reference category (drop_first): >7 → all zeros
        a1c_val              = request.form["A1Cresult"]
        A1Cresult_gt8        = 1 if a1c_val == ">8"          else 0
        A1Cresult_Norm       = 1 if a1c_val == "Norm"        else 0
        A1Cresult_NotMeasured = 1 if a1c_val == "NotMeasured" else 0
        # ">7" → all three stay 0 (reference category)

        # ── max_glu_serum OHE ─────────────────────────────────────────────
        # Reference category (drop_first): >200 → all zeros
        glu_val                    = request.form["max_glu_serum"]
        max_glu_serum_gt300        = 1 if glu_val == ">300"        else 0
        max_glu_serum_Norm         = 1 if glu_val == "Norm"        else 0
        max_glu_serum_NotMeasured  = 1 if glu_val == "NotMeasured" else 0
        # ">200" → all three stay 0 (reference category) — BUG FIX: was missing from original

        # ── Assemble feature DataFrame ────────────────────────────────────
        # Column order matches encode_clf() output exactly
        input_data = pd.DataFrame([{
            "admission_type_id"         : admission_type_id,
            "discharge_disposition_id"  : discharge_disposition_id,
            "admission_source_id"       : admission_source_id,
            "num_lab_procedures"        : num_lab_procedures,
            "num_procedures"            : num_procedures,
            "num_medications"           : num_medications,
            "number_outpatient"         : number_outpatient,
            "number_emergency"          : number_emergency,
            "number_inpatient"          : number_inpatient,
            "number_diagnoses"          : number_diagnoses,
            "insulin"                   : insulin,
            "change"                    : change,
            "diabetesMed"               : diabetesMed,
            "age_numeric"               : age_numeric,
            "total_visits"              : total_visits,
            "total_procedures"          : total_procedures,
            "race_Caucasian"            : race_Caucasian,
            "race_Missing"              : race_Missing,
            "race_Other"                : race_Other,
            "gender_Male"               : gender_Male,
            "A1Cresult_>8"              : A1Cresult_gt8,
            "A1Cresult_Norm"            : A1Cresult_Norm,
            "A1Cresult_NotMeasured"     : A1Cresult_NotMeasured,
            "max_glu_serum_>300"        : max_glu_serum_gt300,
            "max_glu_serum_Norm"        : max_glu_serum_Norm,
            "max_glu_serum_NotMeasured" : max_glu_serum_NotMeasured,
        }])

        # ── Align column order to each model independently ────────────────
        input_clf = input_data[model_readmit.feature_names_in_]
        input_reg = input_data[model_time.feature_names_in_]

        # ── Predictions ───────────────────────────────────────────────────
        # Classification: probability of class 1 (READMITTED)
        # classes_ = [0, 1] → index 1 = READMITTED
        # Threshold 0.3 (instead of 0.5) errs on the side of catching at-risk patients
        prob                  = model_readmit.predict_proba(input_clf)[0][1]
        readmitted_prediction = 1 if prob > 0.3 else 0

        # Regression: clip to realistic range (hospital stay is always ≥ 1 day)
        time_prediction = max(1, round(model_time.predict(input_reg)[0]))

    return render_template(
        "index.html",
        readmitted_prediction=readmitted_prediction,
        time_prediction=time_prediction,
    )


if __name__ == "__main__":
    app.run(debug=True)