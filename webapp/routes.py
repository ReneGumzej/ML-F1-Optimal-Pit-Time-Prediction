from . import app
from flask import render_template,request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


@app.route('/', methods=['GET','POST'])
def home():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data= CustomData(
            event_name=request.form.get('EventName'),
            event_year=request.form.get('eventYear'),
            team=request.form.get('team'),
            compound=request.form.get('compound'),
            driver=request.form.get('driver'),
            best_lap_time_from=request.form.get('bestLapTimeIsFrom'),
            round_number=request.form.get('RoundNumber'),
            stint=request.form.get('stint'),
            best_pre_race_time=request.form.get('bestPreRaceTime'),
            mean_air_temp=request.form.get('meanAirTemp'),
            mean_track_temp=request.form.get('meanTrackTemp'),
            mean_humid=request.form.get('meanHumid'),
            rainfall=request.form.get('Rainfall'),
            grid_position=request.form.get('GridPosition'),
            position=request.form.get('Position'),
            race_stint_nums=request.form.get('raceStintsNums'),
            tyre_age=request.form.get('TyreAge'),
            lap_number_begin_stint=request.form.get('lapNumberAtBeginingOfStint'),
            circiut_lenght=request.form.get('CircuitLength'),
            designed_laps=request.form.get('designedLaps'),
            fuel_slope=request.form.get('fuel_slope'),
            fuel_bias=request.form.get('fuel_bias'),
            deg_slope=request.form.get('deg_slope'),
            deg_bias=request.form.get('deg_bias'),
            lag_slope_mean=request.form.get('lag_slope_mean'),
            lag_bias_mean=request.form.get('lag_bias_mean')
        )
        
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=results[0])