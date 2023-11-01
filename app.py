# app.py

from flask import Flask, render_template, request
import pandas as pd
import torch
#from model import predict
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")   # Use CPU

app = Flask(__name__)

# Define the neural network architecture
class MultiOutputNN(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(MultiOutputNN, self).__init__()
        self.shared_hidden_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        self.output_layers = nn.ModuleList([
            nn.Linear(64, out_dim) for out_dim in output_dims
        ])
                
    def forward(self, x):
        shared_output = self.shared_hidden_layer(x)
        print(f'shared_output shape: {shared_output.shape}')
                                    
        # Before matrix multiplication
        print(f'input shape: {x.shape}')
        
        outputs = [output_layer(shared_output) for output_layer in self.output_layers]
        for i, output in enumerate(outputs):
            print(f'output {i} shape: {output.shape}')
        return outputs



# Define the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Dummy preprocessing function
def preprocess_input(vibration_file, gas_file):
    vibration_data = pd.read_csv(vibration_file)
    gas_data = pd.read_csv(gas_file, sep=';')

    # Perform any necessary preprocessing here
    
    gas_data.drop(['Unnamed: 15', 'Unnamed: 16', 'Date', 'Time', 'NMHC(GT)'], axis=1, inplace=True)
    cleaned_gas_data = gas_data.dropna()
    print(cleaned_gas_data.info())
    def replace_comma_with_period_in_columns(df, columns):
        for column in columns:
            df[column] = df[column].str.replace(',', '.', regex=False)
        return df

    comma_col = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
    cleaned_gas_data = replace_comma_with_period_in_columns(cleaned_gas_data, comma_col)

    for i in comma_col:
        cleaned_gas_data[i] = pd.to_numeric(cleaned_gas_data[i], errors='coerce')

    
    # Concatenate the data
    processed_data = np.hstack((vibration_data.iloc[:9357, :], cleaned_gas_data))

    # Standardize the data
    scaler = StandardScaler()
    processed_data_scaled = pd.DataFrame(scaler.fit_transform(processed_data))
    processed_data_scaled = processed_data_scaled.values

        # Convert processed data to tensor
    X_data_tensor = torch.Tensor(processed_data_scaled ).to(device)

    return X_data_tensor


# this function sends email to representatives if the abnormality is detected 
# in the machine from the vibration sensor or if a gas is detected in environment 
# from the gas sensor

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body):
    sender_email = 'bidehassan@gmail.com' 
    sender_password = 'rmih ytdp znow dgjw'
    recipient_email = 'bidehassan@gmail.com'

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject

    message.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

# Example usage:
# send_email("Anomaly Detected", "Anomalies have been detected in both gas and vibration sensors.")


# gas sensor detection function
def detect_gas_anomaly(gas_sensor):
    thresholds = {
    'CO(GT)': gas_sensor['CO(GT)'].mean() - 2 * gas_sensor['CO(GT)'].std(),
    'PT08.S1(CO)': gas_sensor['PT08.S1(CO)'].mean() - 2 * gas_sensor['PT08.S1(CO)'].std(),
    'C6H6(GT)': gas_sensor['C6H6(GT)'].mean() - 2 * gas_sensor['C6H6(GT)'].std(),
    'PT08.S2(NMHC)': gas_sensor['PT08.S2(NMHC)'].mean() - 2 * gas_sensor['PT08.S2(NMHC)'].std(),
    'NOx(GT)': gas_sensor['NOx(GT)'].mean() - 2 * gas_sensor['NOx(GT)'].std(),
    'PT08.S3(NOx)': gas_sensor['PT08.S3(NOx)'].mean() - 2 * gas_sensor['PT08.S3(NOx)'].std(),
    'NO2(GT)': gas_sensor['NO2(GT)'].mean() - 2 * gas_sensor['NO2(GT)'].std(),
    'PT08.S4(NO2)': gas_sensor['PT08.S4(NO2)'].mean() - 2 * gas_sensor['PT08.S4(NO2)'].std(),
    'PT08.S5(O3)': gas_sensor['PT08.S5(O3)'].mean() - 2 * gas_sensor['PT08.S5(O3)'].std(),
    'T': gas_sensor['T'].mean() - 2 * gas_sensor['T'].std(),
    'RH': gas_sensor['RH'].mean() - 2 * gas_sensor['RH'].std(),
    'AH': gas_sensor['AH'].mean() - 2 * gas_sensor['AH'].std()
}
    
    # # Create a DataFrame to store anomaly flags
    # anomalies = pd.DataFrame(index=gas_data.index)
    
    # for parameter in thresholds.keys():
    #     # Detect anomalies for each parameter
    #     is_anomaly = gas_data[parameter] < thresholds[parameter]
    #     anomalies[f'{parameter}_Anomaly'] = is_anomaly.astype(int)
    
    # return anomalies
    anomalies = []

    for _, data_point in gas_sensor.iterrows():
        data_point_anomaly = {}
        for parameter, threshold in thresholds.items():
            data_point_anomaly[f'{parameter}_Anomaly'] = 1 if data_point[parameter] < threshold else 0
        anomalies.append(data_point_anomaly)

        return anomalies

# vibration sensor abnormality detection function
# def detect_vibration_anomaly(vibration_data):
#     thresholds = {
#         'Vibration_1': 1.226e-1,
#         'Vibration_2': 2.413e-1,
#         'Vibration_3': 1.187e-1
#     }
    
#     # Create a DataFrame to store anomaly flags
#     anomalies = pd.DataFrame(index=range(len(vibration_data)))  # Assuming list of lists
    
#     for i, sensor_readings in enumerate(vibration_data):
#         for sensor, threshold in thresholds.items():
#             # Detect anomalies for each sensor
#             is_anomaly = sensor_readings[i] > threshold
#             anomalies[f'{sensor}_Anomaly'] = is_anomaly.astype(int)
    
#     return anomalies

# def detect_vibration_anomaly(vibration_data):
#     thresholds = {
#         'Vibration_1': 1.226e-1,
#         'Vibration_2': 2.413e-1,
#         'Vibration_3': 1.187e-1
#     }
    
    # Create a list to store anomaly flags
    anomalies = []

    # for sensor_readings in vibration_data:
    #     sensor_anomalies = {}  # Store anomalies for each sensor
    #     for i, (sensor, threshold) in enumerate(thresholds.items()):
    #         # Detect anomalies for each sensor
    #         is_anomaly = sensor_readings[i+2] > threshold  # Assuming sensor data starts from index 2
    #         sensor_anomalies[f'{sensor}_Anomaly'] = int(is_anomaly)
    #     anomalies.append(sensor_anomalies)

    # return anomalies

    # for data_point in vibration_data:
    #     data_point_anomaly = {}
    #     for i, (sensor, threshold) in enumerate(thresholds.items()):
    #         data_point_anomaly[f'{sensor}_Anomaly'] = 1 if data_point[i+2] > threshold else 0
    #     anomalies.append(data_point_anomaly)
    
    #     return anomalies

# def detect_vibration_anomaly(predictions):
#     threshold = -9.8  # Set your threshold value

#     # Create a list to store anomaly flags
#     anomalies = []

#     for prediction in predictions:
#         data_point_anomaly = {}
#         data_point_anomaly['Vibration_1_Anomaly'] = 1 if prediction < threshold else 0
#         data_point_anomaly['Vibration_2_Anomaly'] = 1 if prediction < threshold else 0
#         data_point_anomaly['Vibration_3_Anomaly'] = 1 if prediction < threshold else 0
#         anomalies.append(data_point_anomaly)

#         return anomalies
def detect_vibration_anomaly(vibration_data):
    predicted_values = vibration_data  # Replace with your actual predicted values
    
    # Calculate mean and standard deviation
    mean_value = np.mean(predicted_values)
    std_dev = np.std(predicted_values)
    
    # Define a multiplier (e.g., 2 for 2 standard deviations)
    multiplier = 2
    
    # Calculate threshold
    threshold = mean_value - (multiplier * std_dev)
    
    # Create a list to store anomaly flags
    anomalies = [1 if value < threshold else 0 for value in predicted_values]
    
    return anomalies



    # Use the model to make predictions
def predict(input_data):
    # Load the model
    model = MultiOutputNN(input_dim=17, output_dims=[1, 11])
    model.load_state_dict(torch.load('multi_output_model.pth'))

    model.eval()
    with torch.no_grad():
        outputs = model(input_data)
        print(outputs)
        regression_prediction = outputs[0].cpu().numpy() #item()  # Assuming first output is regression
        classification_prediction = outputs[1].cpu().numpy() #item()  # Assuming second output is classification

    return regression_prediction, classification_prediction


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_files():
    vibration_file = request.files['vibration_data']
    gas_file = request.files['gas_data']

    vibration_path = f"{app.config['UPLOAD_FOLDER']}/vibration.csv"
    print(vibration_path)
    gas_path = f"{app.config['UPLOAD_FOLDER']}/gas.csv"

    vibration_file.save(vibration_path)
    gas_file.save(gas_path)

    # Preprocess the uploaded files
    processed_input = preprocess_input(vibration_path, gas_path)

    # Use the `processed_input` in your predict function
    prediction = predict(processed_input) 
    regression_prediction, classification_prediction = predict(processed_input)  # Include regression prediction

    # Read and process the gas and vibration data
    gas_data = pd.read_csv(gas_path, sep=';')
    gas_data.drop(['Unnamed: 15', 'Unnamed: 16', 'Date', 'Time', 'NMHC(GT)'], axis=1, inplace=True)

    def replace_comma_with_period_in_columns(df, columns):
        for column in columns:
            df[column] = df[column].str.replace(',', '.', regex=False)
        return df

    comma_col = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
    gas_data = replace_comma_with_period_in_columns(gas_data, comma_col)

    for i in comma_col:
        gas_data[i] = pd.to_numeric(gas_data[i], errors='coerce')
    
    gas_anomaly =  gas_data.copy() # have a copy of the dataframe before converted tp list
    gas_data = gas_data.values.tolist()

    vibration_data = pd.read_csv(vibration_path).values[:9357, :].tolist()
    vib_data = pd.read_csv(vibration_path).values[:9357, :]

     # Detect anomalies in vibration data
    anomalies_vibration = detect_vibration_anomaly(prediction[0])

    # Detect anomalies in gas data
    gas_sensor = pd.DataFrame(gas_data, columns=['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'])
    anomalies_gas = detect_gas_anomaly(gas_sensor)

    # Send email if anomalies are detected
    if anomalies_vibration or anomalies_gas:
        send_email("Anomaly Detected", "Anomaly detected in the system!")

    return render_template('result.html', prediction=prediction, gas_data=gas_data, vibration_data=vibration_data, anomalies_vibration=anomalies_vibration, anomalies_gas=anomalies_gas)

if __name__ == '__main__':
    app.run(debug=True)
