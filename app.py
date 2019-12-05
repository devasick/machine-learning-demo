import flask
import pandas as pd
import numpy as np
import json,urllib.request
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


 



def create_app():
    app = flask.Flask(__name__)
     
    @app.route('/', methods=['GET', 'POST'])
    def index(chartID = 'chart_ID', chart_type = 'bar', chart_height = 500):
         

        ############
        data = urllib.request.urlopen("http://0.0.0.0:7411/data").read() 

        output = json.loads(data) #json data fetching from localhost

        data_map = pd.Series(output['sensor_data'],index=output['sensor_data2']).sort_values(ascending=False)
         
        x_value = data_map.index.to_numpy() # index value convert object to array
        sensor_data = list(data_map.values) #data convert to array
         

        ########## start ######3
        
        chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
        series = [ {"name": 'Sensor', "data": sensor_data}]
        title = {"text": 'Important Features'}
        xAxis = {"categories": list(x_value)}
        yAxis = {"title": {"text": 'Feature Importance Score'}}
     
        return flask.render_template('index.html', chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)

    @app.route('/data', methods=['GET', 'POST'])
    def data():
        
        data = pd.read_csv('task_data.csv')
        sample = data.drop(['class_label','sample index'],axis=1)
        label = data['class_label'].map({-1: 0, 1: 1})
        label.value_counts()
        y = label
        X = sample
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        classifier=RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train,y_train)
        y_pred=classifier.predict(X_test)
       
         
        context = {
            'sensor_data': list(classifier.feature_importances_),
            'sensor_data2':list(sample.columns)
        }

        return flask.jsonify(context)

    

    return app




if __name__ == "__main__":
    app = create_app()
    # serve the application on port 7410
    app.run(debug=True,host='0.0.0.0', port=7411)
