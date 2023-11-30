import joblib
import numpy as np
import pandas as pd

data = {
    'left_elbow': [179.25588],
    'left_armpit': [141.9942569],
    'left_waist': [99.80349636],
    'left_knee': [169.4623463],
    'right_elbow': [177.4194415],
    'right_armpit': [146.6215407],
    'right_waist': [101.3268058],
    'right_knee': [173.6137765]
}

df = pd.DataFrame(data)

svc_model = joblib.load('./jobmodel/svc_yoga.joblib')
pred = svc_model.predict(df)
print(pred)



