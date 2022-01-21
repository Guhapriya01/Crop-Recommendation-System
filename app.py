import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
model = pickle.load(open('model/crop_rec_model.pkl', 'rb'))

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        ph = flask.request.form['ph']
        rainfall = flask.request.form['rainfall']
        n = flask.request.form['n']
        p = flask.request.form['p']
        k = flask.request.form['k']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[n,p,k,temperature, humidity,ph,rainfall ]],
                                       columns=['n','p','k','temperature', 'humidity', 'ph','rainfall'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
        if prediction==0:
            prediction='Apple'
        elif prediction==1:
            prediction='Banana'
        elif prediction==2:
            prediction='Blackgram'
        elif prediction==3:
            prediction='Chickpea'
        elif prediction==4:
            prediction='Coconut'
        elif prediction==5:
            prediction='Coffee'
        elif prediction==6:
            prediction='Jute'
        elif prediction==7:
            prediction='Grapes'
        elif prediction==8:
            prediction='Cotton'
        elif prediction==9:
            prediction='Kidneybeans'
        elif prediction==10:
            prediction='Lentil'
        elif prediction==11:
            prediction='Maize'
        elif prediction==12:
            prediction='Mango'
        elif prediction==13:
            prediction='Mothbeans'
        elif prediction==14:
            prediction='Mungbean'
        elif prediction==15:
            prediction='MuskMelon'
        elif prediction==16:
            prediction='Orange'
        elif prediction==17:
            prediction='Papaya'
        elif prediction==18:
            prediction='Pigeonpeas'
        elif prediction==19:
            prediction='Pomegranate'
        elif prediction==20:
            prediction='Rice'
        elif prediction==21:
            prediction='WaterMelon'
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'N':n,
                                                     'P':p,
                                                     'K':k,
                                                     'Temperature':temperature,
                                                     'Humidity':humidity,
                                                     'Ph':ph,
                                                     'Rainfall':rainfall},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()