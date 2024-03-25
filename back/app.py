from flask import Flask, redirect, url_for, request
import pandas as pd
import joblib

app = Flask(__name__)

features = ['Диаметр экспл.колонны', 'Буферное давление', 'Давление в линии', 'Динамическая высота', 'Затрубное давление', 'Обводненность', 'Глубина спуска']
modelsNames = ['Прогноз (регрессия)','Прогноз (случайный лес)', 'Прогноз (градиентный бустинг)']
modelsPath = {
    'Прогноз (регрессия)': '../modeling/models/best_model_reg.pkl',
    'Прогноз (случайный лес)': '../modeling/models/best_model_rf.pkl',
    'Прогноз (градиентный бустинг)': '../modeling/models/best_model_rf.pkl'
}

@app.route('/success/<reqDict>')
def success(reqDict):
    styleStr = """
    * {
        background-color: #0070b7;
        color: white;
        font-size: 16px;
        font-weight: 100;
        font-family: Helvetica;
    }

    input {
        background-color: white;
        border-radius:10px;
        color: #0070b7;
    }

    p {
        font-weight: 600;
        font-size: 20px;
    }
    """
    paramsStr = '<style>'+styleStr+'</style>'
    reqDict = eval(reqDict)
    for feature in list(reqDict.keys()):
        paramsStr += '<p>' + feature + ':</p> %s\n' % reqDict[feature]
    return paramsStr

@app.route('/predict', methods = ['POST'])
def login():
    if request.method == 'POST':
        reqDict = {}
        for feature in features:
            reqDict[feature] = request.form[feature]
            
            
        x = pd.DataFrame(reqDict, index=[0])
        for modelsName in modelsNames:
            modelPath = modelsPath[modelsName]
            model = joblib.load(modelPath)
            y = model.predict(x)
            reqDict[modelsName] = y[0]

        
        return redirect(url_for('success', reqDict=reqDict))

if __name__ == '__main__':
    app.run(debug = True)