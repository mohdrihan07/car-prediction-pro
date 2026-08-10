
from flask import Flask, request, render_template_string
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)

data = {
    "Car_Model": [
        "Swift", "Swift", "Swift", "Swift",
        "Baleno", "Baleno", "Baleno", "Baleno",
        "Creta", "Creta", "Creta", "Creta",
        "i20", "i20", "i20", "i20",
        "Nexon", "Nexon", "Nexon", "Nexon"
    ],

    "Year": [
        2018, 2019, 2020, 2021,
        2018, 2019, 2020, 2021,
        2018, 2019, 2020, 2021,
        2018, 2019, 2020, 2021,
        2018, 2019, 2020, 2021
    ],

    "KMs_Driven": [
        60000, 45000, 35000, 20000,
        55000, 40000, 30000, 18000,
        70000, 50000, 30000, 18000,
        60000, 40000, 25000, 15000,
        65000, 45000, 30000, 15000
    ],

    "Fuel_Type": [
        "Petrol", "Petrol", "Petrol", "Petrol",
        "Petrol", "Petrol", "Petrol", "Petrol",
        "Diesel", "Diesel", "Diesel", "Diesel",
        "Petrol", "Petrol", "Petrol", "Petrol",
        "Petrol", "Petrol", "Petrol", "Petrol"
    ],

    "Transmission": [
        "Manual", "Manual", "Manual", "Automatic",
        "Manual", "Manual", "Automatic", "Automatic",
        "Manual", "Manual", "Automatic", "Automatic",
        "Manual", "Manual", "Automatic", "Automatic",
        "Manual", "Manual", "Automatic", "Automatic"
    ],

    "Owner": [
        2, 1, 1, 1,
        2, 1, 1, 1,
        2, 1, 1, 1,
        2, 1, 1, 1,
        2, 1, 1, 1
    ],

    # Used car price in Lakh
    "Price": [
        4.0, 4.5, 5.2, 6.0,
        4.5, 5.0, 6.0, 6.8,
        8.0, 9.0, 11.0, 13.0,
        4.2, 4.8, 5.8, 6.5,
        5.0, 5.8, 6.8, 7.8
    ]
}



df = pd.DataFrame(data)


X = df[
    [
        "Car_Model",
        "Year",
        "KMs_Driven",
        "Fuel_Type",
        "Transmission",
        "Owner"
    ]
]

y = df["Price"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



categorical_columns = [
    "Car_Model",
    "Fuel_Type",
    "Transmission"
]



preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)



model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)



model.fit(X_train, y_train)



y_pred = model.predict(X_test)

accuracy = r2_score(y_test, y_pred) * 100



HTML = """

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Car Price Predictor</title>


<style>

/* ==============================
   GLOBAL
================================ */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: Arial, sans-serif;
}


body {

    min-height: 100vh;

    background:
    linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a,
        #312e81
    );

    display: flex;

    justify-content: center;

    align-items: center;

    padding: 30px;

}


/* ==============================
   MAIN CARD
================================ */

.container {

    width: 900px;

    max-width: 100%;

    background: white;

    border-radius: 25px;

    overflow: hidden;

    box-shadow:
    0 25px 60px
    rgba(0,0,0,0.35);

}


/* ==============================
   HEADER
================================ */

.header {

    padding: 40px;

    text-align: center;

    color: white;

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

}


.car-icon {

    font-size: 55px;

    margin-bottom: 10px;

}


.header h1 {

    font-size: 34px;

    margin-bottom: 8px;

}


.header p {

    font-size: 16px;

    opacity: 0.9;

}


/* ==============================
   FORM
================================ */

.form-section {

    padding: 40px;

}


.form-title {

    font-size: 22px;

    font-weight: bold;

    color: #1e293b;

    margin-bottom: 25px;

}


.form-grid {

    display: grid;

    grid-template-columns:
    1fr 1fr;

    gap: 22px;

}


.input-box {

    display: flex;

    flex-direction: column;

}


label {

    font-weight: bold;

    color: #334155;

    margin-bottom: 8px;

}


input,
select {

    padding: 14px;

    border: 1px solid #cbd5e1;

    border-radius: 10px;

    font-size: 15px;

    background: #f8fafc;

    outline: none;

}


input:focus,
select:focus {

    border-color: #2563eb;

    background: white;

}


/* ==============================
   BUTTON
================================ */

.predict-btn {

    width: 100%;

    margin-top: 30px;

    padding: 16px;

    border: none;

    border-radius: 12px;

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    color: white;

    font-size: 17px;

    font-weight: bold;

    cursor: pointer;

}


.predict-btn:hover {

    transform: translateY(-2px);

    box-shadow:
    0 10px 25px
    rgba(37,99,235,0.3);

}


/* ==============================
   RESULT
================================ */

.result {

    margin-top: 30px;

    padding: 25px;

    border-radius: 15px;

    text-align: center;

    background: #eff6ff;

    border: 1px solid #bfdbfe;

}


.result-title {

    color: #475569;

    font-size: 16px;

    margin-bottom: 8px;

}


.price {

    font-size: 38px;

    font-weight: bold;

    color: #2563eb;

}


.result-text {

    margin-top: 8px;

    color: #64748b;

}


/* ==============================
   MODEL INFO
================================ */

.model-info {

    margin-top: 25px;

    padding: 18px;

    border-radius: 12px;

    background: #f8fafc;

    text-align: center;

    color: #475569;

}


.model-info strong {

    color: #2563eb;

}



.footer {

    text-align: center;

    padding: 20px;

    color: #64748b;

    font-size: 13px;

}



@media(max-width: 650px) {

    .form-grid {

        grid-template-columns: 1fr;

    }

    .header h1 {

        font-size: 27px;

    }

    .form-section {

        padding: 25px;

    }

}

</style>

</head>


<body>


<div class="container">


<!-- HEADER -->

<div class="header">

    <div class="car-icon">
        
    </div>

    <h1>
        Car Price Predictor
    </h1>

    <p>
        Machine Learning Based Used Car Price Prediction
    </p>

</div>



<!-- FORM -->

<div class="form-section">


<div class="form-title">

    Enter Car Details

</div>


<form method="POST">


<div class="form-grid">


<!-- CAR MODEL -->

<div class="input-box">

<label>
Car Model
</label>

<select name="car_model" required>

<option value="">
Select Car Model
</option>

<option value="Swift">
Maruti Swift
</option>

<option value="Baleno">
Maruti Baleno
</option>

<option value="Creta">
Hyundai Creta
</option>

<option value="i20">
Hyundai i20
</option>

<option value="Nexon">
Tata Nexon
</option>

</select>

</div>



<!-- YEAR -->

<div class="input-box">

<label>
Manufacturing Year
</label>

<input
type="number"
name="year"
placeholder="Example: 2020"
min="2000"
max="2026"
required>

</div>



<!-- KM -->

<div class="input-box">

<label>
Kilometers Driven
</label>

<input
type="number"
name="kms"
placeholder="Example: 35000"
min="0"
required>

</div>



<!-- FUEL -->

<div class="input-box">

<label>
Fuel Type
</label>

<select name="fuel" required>

<option value="">
Select Fuel
</option>

<option value="Petrol">
Petrol
</option>

<option value="Diesel">
Diesel
</option>

</select>

</div>



<!-- TRANSMISSION -->

<div class="input-box">

<label>
Transmission
</label>

<select name="transmission" required>

<option value="">
Select Transmission
</option>

<option value="Manual">
Manual
</option>

<option value="Automatic">
Automatic
</option>

</select>

</div>



<!-- OWNER -->

<div class="input-box">

<label>
Previous Owners
</label>

<input
type="number"
name="owner"
placeholder="Example: 1"
min="0"
max="5"
required>

</div>


</div>


<!-- BUTTON -->

<button
type="submit"
class="predict-btn">

 Predict Car Price

</button>


</form>



<!-- RESULT -->

{% if prediction %}

<div class="result">

<div class="result-title">

Estimated Market Value

</div>

<div class="price">

Rs. {{ prediction }} Lakh

</div>

<div class="result-text">

Based on the details you entered

</div>

</div>

{% endif %}



<!-- MODEL INFO -->

<div class="model-info">

 ML Model:

<strong>
Random Forest Regression
</strong>

<br><br>

Model R² Score:

<strong>
{{ accuracy }}%
</strong>

</div>


</div>


<!-- FOOTER -->

<div class="footer">

Car Price Prediction System
| Python + Flask + Machine Learning

</div>


</div>


</body>

</html>

"""


# ============================================================
# 12. HOME + PREDICTION
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        try:

            # Get values from HTML

            car_model = request.form["car_model"]

            year = int(request.form["year"])

            kms = int(request.form["kms"])

            fuel = request.form["fuel"]

            transmission = request.form["transmission"]

            owner = int(request.form["owner"])


            # Create DataFrame

            new_car = pd.DataFrame({

                "Car_Model": [car_model],

                "Year": [year],

                "KMs_Driven": [kms],

                "Fuel_Type": [fuel],

                "Transmission": [transmission],

                "Owner": [owner]

            })


            # Prediction

            prediction = model.predict(new_car)[0]

            prediction = round(prediction, 2)


        except Exception as e:

            prediction = None

            print("Error:", e)


    return render_template_string(

        HTML,

        prediction=prediction,

        accuracy=round(accuracy, 2)

    )



if __name__ == "__main__":

    print("\n======================================")

    print(" CAR PRICE PREDICTOR")

    print("======================================")


    print("\nOpen this URL in your browser:")

    print("http://127.0.0.1:5000")

    print("======================================\n")

    app.run(debug=True)
    