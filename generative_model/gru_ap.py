import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt
import gzip

class GRUModel(nn.Module):
    def __init__(self, static_features, sequence_features, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.static_fc = nn.Linear(static_features, hidden_size)
        self.gru = nn.GRU(sequence_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, static, sequence):
        static_out = self.static_fc(static)
        _, gru_out = self.gru(sequence)
        combined = torch.cat((static_out, gru_out.squeeze(0)), dim=1)
        out = self.fc(combined)
        return self.sigmoid(out)

def prepare_data(data, sequence_length, medical_factors):
    sequences = []
    static_features = []
    targets = []
    
    for patient_id in data['SUBJECT_ID'].unique():
        patient_data = data[data['SUBJECT_ID'] == patient_id].sort_values('days_between_visits')
        if len(patient_data) > sequence_length:
            for i in range(len(patient_data) - sequence_length):
                seq = patient_data.iloc[i:i+sequence_length]
                next_visit = patient_data.iloc[i+sequence_length]
                
                static = [
                    next_visit['Age_max'],
                    next_visit['GENDER_M'],
                    next_visit['GENDER_F'],
                    next_visit['days_between_visits'],
                    next_visit['RELIGION_CATHOLIC'],
             next_visit['RELIGION_Otra'],
             next_visit['RELIGION_Unknown'],
             next_visit['MARITAL_STATUS_0'],
             next_visit['MARITAL_STATUS_DIVORCED'],
             next_visit['MARITAL_STATUS_LIFE PARTNER'],
             next_visit['MARITAL_STATUS_MARRIED'],
             next_visit['MARITAL_STATUS_SEPARATED'],
             next_visit['MARITAL_STATUS_SINGLE'],
             next_visit['MARITAL_STATUS_Unknown'],
             next_visit['MARITAL_STATUS_WIDOWED'],
             next_visit['ETHNICITY_Otra'],
             next_visit['ETHNICITY_Unknown'],
             next_visit['ETHNICITY_WHITE']
                ]
    
                
                sequence_features = seq[medical_factors].values
                target = next_visit[medical_factors].values
                
                static_features.append(static)
                sequences.append(sequence_features)
                targets.append(target)
    
    return np.array(static_features), np.array(sequences), np.array(targets)

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for static, sequence, target in train_loader:
            static, sequence, target = static.to(device), sequence.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(static, sequence)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for static, sequence, target in val_loader:
                static, sequence, target = static.to(device), sequence.to(device), target.to(device)
                outputs = model(static, sequence)
                loss = criterion(outputs, target)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def calculate_log_likelihood(model, data_loader, device):
    model.eval()
    total_log_likelihood = 0
    with torch.no_grad():
        for static, sequence, target in data_loader:
            static, sequence, target = static.to(device), sequence.to(device), target.to(device)
            outputs = model(static, sequence)
            log_likelihood = torch.sum(target * torch.log(outputs + 1e-10) + (1 - target) * torch.log(1 - outputs + 1e-10))
            total_log_likelihood += log_likelihood.item()
    return total_log_likelihood


def load_data(file_path):
    try:
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        with open(file_path, 'rb') as f:
             data = pickle.load(f)    
             return data

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)    

def convert_to_binary(data, columns):
    for col in columns:
        data[col] = (data[col] > 0).astype(int)
    return data
# Main execution



save_path_arf = "ARF/GRU/"
data = load_data("D:\Synthetic-Data-Deep-Learning\data\intermedi\SD\inpput\entire_ceros_tabular_data.pkl")



columns_to_drop = ['GENDER_0', 'ADMITTIME']
data = data.drop(columns=columns_to_drop)

drug_columns = list(data.filter(like="drugs").columns)
diagnosis_columns = list(data.filter(like="diagnosis").columns)
procedure_columns = list(data.filter(like="procedures").columns)
medical_factors = drug_columns + diagnosis_columns + procedure_columns

# Convert medical factors to binary
data = convert_to_binary(data, medical_factors)

# Prepare data
sequence_length = 5  # Adjust as needed
static_features, sequences, targets = prepare_data(data, sequence_length, medical_factors)

# Normalize static features
scaler = StandardScaler()
static_features_scaled = scaler.fit_transform(static_features)

# Split data into train, validation, and test sets
static_train_val, static_test, seq_train_val, seq_test, y_train_val, y_test = train_test_split(
    static_features_scaled, sequences, targets, test_size=0.2, random_state=42)

static_train, static_val, seq_train, seq_val, y_train, y_val = train_test_split(
    static_train_val, seq_train_val, y_train_val, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and create DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(static_train), torch.FloatTensor(seq_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(static_val), torch.FloatTensor(seq_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(static_test), torch.FloatTensor(seq_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
static_features = static_train.shape[1]
sequence_features = sequences.shape[2]
hidden_size = 64  # Adjust as needed
output_size = len(medical_factors)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRUModel(static_features, sequence_features, hidden_size, output_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 50  # Adjust as needed
train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(save_path_arf + 'loss_plot.png')
plt.close()

# Evaluate on test set
test_loss = calculate_log_likelihood(model, test_loader, device)
print(f"Test Log-Likelihood: {test_loss:.4f}")

# Save the model and scaler
torch.save(model.state_dict(), save_path_arf + 'gru_conditional_probability_model.pth')
with open(save_path_arf + 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Example inference
sample_static = np.array([[
    65,  # Age_max
    1,   # GENDER_M
    0,   # GENDER_F
    30,  # days_between_visits
    1,   # RELIGION_CATHOLIC
    0,   # RELIGION_Otra
    0,   # RELIGION_Unknown
    0,   # MARITAL_STATUS_0
    0,   # MARITAL_STATUS_DIVORCED
    0,   # MARITAL_STATUS_LIFE PARTNER
    1,   # MARITAL_STATUS_MARRIED
    0,   # MARITAL_STATUS_SEPARATED
    0,   # MARITAL_STATUS_SINGLE
    0,   # MARITAL_STATUS_Unknown
    0,   # MARITAL_STATUS_WIDOWED
    0,   # ETHNICITY_Otra
    0,   # ETHNICITY_Unknown
    1    # ETHNICITY_WHITE
]])  # Age, GENDER_M, GENDER_F, days_between_visits
sample_static_scaled = scaler.transform(sample_static)
sample_sequence = np.zeros((1, sequence_length, len(medical_factors)))  # Assuming no previous medical history

sample_static_tensor = torch.FloatTensor(sample_static_scaled).to(device)
sample_sequence_tensor = torch.FloatTensor(sample_sequence).to(device)

model.eval()
with torch.no_grad():
    sample_probabilities = model(sample_static_tensor, sample_sequence_tensor).cpu().numpy()[0]

print("\nConditional Probabilities for Sample Patient:")
print("Patient characteristics: Age=65, Male, 30 days since last visit")
for factor, prob in zip(medical_factors, sample_probabilities):
    print(f"{factor}: P({factor} | patient characteristics) = {prob:.4f}")