from flask import Flask, request, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


main_model = load_pickle("rf_main_model.pkl")
main_features = load_pickle("features_main_model.pkl")

efficiency_model = load_pickle("rf_efficiency_model.pkl")
efficiency_features = load_pickle("features_efficiency_model.pkl")


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Olympic Medal Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 850px;
            margin: 40px auto;
            background: #f4f6f8;
            color: #222;
        }

        h1 {
            text-align: center;
        }

        .card {
            background: white;
            padding: 25px;
            margin: 25px 0;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        }

        label {
            font-weight: bold;
        }

        input {
            width: 100%;
            padding: 8px;
            margin-top: 6px;
            margin-bottom: 18px;
        }

        button {
            padding: 10px 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }

        .result {
            background: #e9ffe9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
    </style>
</head>

<body>
    <h1>Olympic Medal Predictor</h1>

    <div class="card">
        <h2>Main Model</h2>
        <p>Uses previous medals and athletes sent.</p>

        <form action="/predict_main" method="post">
            <label>Previous Total Medals</label>
            <input name="prev_total_medals" type="number" step="any" required>

            <label>Athletes Sent</label>
            <input name="athletes_sent" type="number" step="any" required>

            <button type="submit">Predict with Main Model</button>
        </form>

        {% if main_prediction is not none %}
            <div class="result">
                <h3>Main Model Prediction: {{ main_prediction }} medals</h3>
            </div>
        {% endif %}
    </div>

    <div class="card">
        <h2>Efficiency Model</h2>
        <p>Uses GDP per capita, population, and HDI.</p>

        <form action="/predict_efficiency" method="post">
            <label>GDP per Capita</label>
            <input name="gdp_per_capita" type="number" step="any" required>

            <label>Population</label>
            <input name="population" type="number" step="any" required>

            <label>Human Development Index</label>
            <input name="hdi" type="number" step="any" min="0" max="1" required>

            <button type="submit">Predict with Efficiency Model</button>
        </form>

        {% if efficiency_prediction is not none %}
            <div class="result">
                <h3>Efficiency Model Prediction: {{ efficiency_prediction }} medals</h3>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(
        HTML,
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

    return render_template_string(
        HTML,
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

    return render_template_string(
        HTML,
        main_prediction=None,
        efficiency_prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)