import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_rnn    = joblib.load(os.path.join(BASE_DIR, "rnn_model.joblib"))
_scaler = joblib.load(os.path.join(BASE_DIR, "rnn_scaler.joblib"))


def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    raw    = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled = _scaler.transform(raw)
    seq    = scaled.reshape(1, 5, 1)

    preds, probas = _rnn.predict(seq)
    pred      = int(preds[0])
    prob_pass = float(probas[0]) * 100

    if prob_pass >= 80:
        performance = "High 🌟"
    elif prob_pass >= 50:
        performance = "Medium ⚠️"
    else:
        performance = "Low 🔴"

    return {
        "prediction" : pred,
        "label"      : "Pass ✅" if pred == 1 else "Fail ❌",
        "prob_pass"  : round(prob_pass, 1),
        "prob_fail"  : round(100 - prob_pass, 1),
        "performance": performance,
    }
