from flask import Flask, request, render_template
from src.pipeline.predict import CustomData, PredictPipeline 

application=Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

def clean_input(input_data):
    # If the input is a tuple, return the first element
    if isinstance(input_data, tuple):
        return input_data[0]  # Only return the first element of the tuple
    # If the input has extra spaces or unwanted characters, clean them
    return input_data.strip() if input_data else ""


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:

        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=clean_input(request.form.get('race_ethnicity')),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score')),
        ) 
        race = request.form.get('race_ethnicity')
        print("race ethnicity after the form", race)
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
        

if __name__=="__main__":
    app.run(host='0.0.0.0')