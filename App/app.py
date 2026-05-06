from flask import Flask, request, render_template
import pandas as pd
import pickle
from pathlib import Path

app = Flask(__name__)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)




BASE_DIR = Path(__file__).resolve().parent

main_model = load_pickle(BASE_DIR / "rf_main_model.pkl")
main_features = load_pickle(BASE_DIR / "features_main_model.pkl")

efficiency_model = load_pickle(BASE_DIR / "rf_efficiency_model.pkl")
efficiency_features = load_pickle(BASE_DIR / "features_efficiency_model.pkl")


@app.route("/")
def home():
    return render_template(
        "index.html",
        main_prediction=None,
        efficiency_prediction=None
    )


@app.route("/predict_main", methods=["POST"])
def predict_main():
    data = {
        "prev_total_medals": float(request.form["prev_total_medals"]),
        "athletes_sent": float(request.form["athletes_sent"])
    }

    X = pd.DataFrame([data])[main_features]
    prediction = main_model.predict(X)[0]

    return render_template(
        "index.html",
        main_prediction=round(prediction, 2),
        efficiency_prediction=None
    )


@app.route("/predict_efficiency", methods=["POST"])
def predict_efficiency():
    data = {
        "gdp_per_capita": float(request.form["gdp_per_capita"]),
        "population": float(request.form["population"]),
        "hdi": float(request.form["hdi"])
    }

    X = pd.DataFrame([data])[efficiency_features]
    prediction = efficiency_model.predict(X)[0]

    return render_template(
        "index.html",
        main_prediction=None,
        efficiency_prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)