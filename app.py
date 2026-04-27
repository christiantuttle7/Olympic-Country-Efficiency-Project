from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

# Main model: prev_total_medals + athletes_sent
with open("rf_main_model.pkl", "rb") as f:
    main_model = pickle.load(f)

with open("features_main_model.pkl", "rb") as f:
    main_features = pickle.load(f)

# HDI model: gdp_per_capita + population + hdi
with open("rf_efficiency_model.pkl", "rb") as f:
    hdi_model = pickle.load(f)

with open("features_efficiency_model.pkl", "rb") as f:
    hdi_features = pickle.load(f)


@app.route("/")
def home():
    return """
    <h1>Olympic Medal Predictor</h1>

    <h2>Main Model</h2>
    <form action="/predict_main" method="post">
        Previous Total Medals:
        <input name="prev_total_medals"><br><br>

        Athletes Sent:
        <input name="athletes_sent"><br><br>

        <button type="submit">Predict with Main Model</button>
    </form>

    <hr>

    <h2>Efficiency Model</h2>
    <form action="/predict_hdi" method="post">
        GDP per Capita:
        <input name="gdp_per_capita"><br><br>

        Population:
        <input name="Population"><br><br>

        Human Development Index:
        <input name="Human Development Index:"><br><br>

        <button type="submit">Predict with HDI Model</button>
    </form>
    """


@app.route("/predict_main", methods=["POST"])
def predict_main():
    data = {
        "prev_total_medals": float(request.form["prev_total_medals"]),
        "athletes_sent": float(request.form["athletes_sent"])
    }

    X = pd.DataFrame([data])[main_features]
    prediction = main_model.predict(X)[0]

    return f"<h1>Main Model Predicted Medals: {round(prediction, 2)}</h1>"


@app.route("/predict_hdi", methods=["POST"])
def predict_hdi():
    data = {
        "gdp_per_capita": float(request.form["gdp_per_capita"]),
        "population": float(request.form["population"]),
        "hdi": float(request.form["hdi"])
    }

    X = pd.DataFrame([data])[hdi_features]
    prediction = hdi_model.predict(X)[0]

    return f"<h1>HDI Model Predicted Medals: {round(prediction, 2)}</h1>"


if __name__ == "__main__":
    app.run(debug=True)