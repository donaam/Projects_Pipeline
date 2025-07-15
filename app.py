from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__, static_folder='templates')
app = application


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        try:
            data = CustomData(
                cc_num=float(request.form.get('cc_num')),
                amt=float(request.form.get('amt')),
                zip=int(request.form.get('zip')),
                lat=float(request.form.get('lat')),
                long=float(request.form.get('long')),
                city_pop=int(request.form.get('city_pop')),
                unix_time=int(request.form.get('unix_time')),
                merch_lat=float(request.form.get('merch_lat')),
                merch_long=float(request.form.get('merch_long')),
                user_transaction_count=int(
                    request.form.get('user_transaction_count')),
                merchant_transaction_count=int(
                    request.form.get('merchant_transaction_count')),
                merchant=request.form.get('merchant'),
                category=request.form.get('category'),
                first=request.form.get('first'),
                last=request.form.get('last'),
                gender=request.form.get('gender'),
                street=request.form.get('street'),
                city=request.form.get('city'),
                state=request.form.get('state'),
                job=request.form.get('job'),
                dob=request.form.get('dob'),
                date=request.form.get('date'),
                time=request.form.get('time')
            )

            new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(new_data)

            result_label = "Fraudulent Transaction"
            if pred[0] == 1:
                result_label = "Fraudulent Transaction"
            else:
                result_label = "Legitimate Transaction"

            return render_template("form.html", final_result=result_label)

        except Exception as e:
            return render_template(
                "form.html", final_result=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
