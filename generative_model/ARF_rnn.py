import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
os.chdir("/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/")
import sys
sys.path.append('/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning/')
from evaluation.resemb.resemblance.EvaluationResemblance import *
from evaluation.resemb.resemblance.utilsstats import *

from evaluation.resemb.resemblance.config_fored import *
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
import scipy
from arfpy import utils

class Config:
    def __init__(self,X):
        self.vocab_dim = X.shape[1]  # Set this to the number of unique medical codes
        self.embedding_dim = 128
        self.num_head = 4
        self.ff_dim = 256
        self.lstm_dim = 128
        self.n_layer = 2
        self.condition_dim = 64

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.code_embed = nn.Embedding(config.vocab_dim+1, config.embedding_dim)
    
    def forward(self, codes):
        return self.code_embed(codes)

class SingleVisitTransformer(nn.Module):
    def __init__(self, config):
        super(SingleVisitTransformer, self).__init__()
        encoderLayer = nn.TransformerEncoderLayer(config.embedding_dim, config.num_head, 
                        dim_feedforward=config.ff_dim, dropout=0.1, activation="relu", 
                        layer_norm_eps=1e-08, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoderLayer, 2)
    
    def forward(self, code_embeddings, visit_lengths):
        bs, vs, cs, ed = code_embeddings.shape
        mask = torch.ones((bs, vs, cs)).to(code_embeddings.device)
        for i in range(bs):
            for j in range(vs):
                mask[i,j,:visit_lengths[i,j]] = 0
        visits = torch.reshape(code_embeddings, (bs*vs,cs,ed))
        mask = torch.reshape(mask, (bs*vs,cs))
        encodings = self.transformer(visits, src_key_padding_mask=mask)
        encodings = torch.reshape(encodings, (bs,vs,cs,ed))
        return encodings[:,:,0,:]

class RecurrentLayer(nn.Module):
    def __init__(self, config):
        super(RecurrentLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=config.lstm_dim, hidden_size=config.lstm_dim, 
                            num_layers=config.n_layer, dropout=0.1)
    
    def forward(self, visit_embeddings):   
        output, _ = self.lstm(visit_embeddings)
        return output

class DependencyModel(nn.Module):
    def __init__(self, config):
        super(DependencyModel, self).__init__()
        self.embeddings = Embedding(config)
        self.visit_att = SingleVisitTransformer(config)
        self.proj1 = nn.Linear(config.embedding_dim, config.lstm_dim)
        self.lstm = RecurrentLayer(config)
        self.proj2 = nn.Linear(config.lstm_dim, config.condition_dim)
        self.proj3 = nn.Linear(config.condition_dim, config.vocab_dim)
        self.proj_days = nn.Linear(config.condition_dim, 1)  # New projection for days prediction
        
    def forward(self, inputs_word, visit_lengths, days_from_last_admission, export=False):
        inputs = self.embeddings(inputs_word)
        inputs = self.visit_att(inputs, visit_lengths)
        inputs = self.proj1(inputs)
        
        # Concatenate days_from_last_admission to the input
        days_embedded = days_from_last_admission.unsqueeze(-1).expand(-1, -1, inputs.size(-1))
        inputs = torch.cat([inputs, days_embedded], dim=-1)
        
        output = self.lstm(inputs)
        condition = self.proj2(output)
        
        if export:
            return condition
        else:
            codes_output = self.proj3(torch.relu(condition))
            days_output = self.proj_days(torch.relu(condition)).squeeze(-1)
            
            sig = nn.Sigmoid()
            diagnosis_output = sig(codes_output[:, :-1, :])
            days_output = torch.relu(days_output[:, :-1])  # Ensure non-negative days
            
            return diagnosis_output, days_output

class EnhancedForge:
    def __init__(self, original_forge, dependency_model, config):
        self.original_forge = original_forge
        self.dependency_model = dependency_model
        self.config = config
        self.admissions_distribution = None

    def prepare_sequence(self, patient_data):
        code_cols = [col for col in patient_data.columns if col.startswith('code_')]
        codes = patient_data[code_cols].values
        codes_tensor = torch.LongTensor(codes)
        visit_lengths = (codes_tensor != 0).sum(dim=-1)
        days_tensor = torch.FloatTensor(patient_data['days_from_last_admission'].values)
        return codes_tensor, visit_lengths, days_tensor

    def train_model(self, real_data,num_epochs):
        grouped_data = real_data.groupby('patient_id')
        self.admissions_distribution = Counter(grouped_data.size())
        
        all_codes, all_lengths, all_days = [], [], []
        for _, group in grouped_data:
            codes, lengths, days = self.prepare_sequence(group)
            all_codes.append(codes)
            all_lengths.append(lengths)
            all_days.append(days)
        
        max_visits = max(len(c) for c in all_codes)
        padded_codes = torch.nn.utils.rnn.pad_sequence(all_codes, batch_first=True)
        padded_lengths = torch.nn.utils.rnn.pad_sequence(all_lengths, batch_first=True)
        padded_days = torch.nn.utils.rnn.pad_sequence(all_days, batch_first=True)
        
        optimizer = torch.optim.Adam(self.dependency_model.parameters())
        criterion_codes = nn.BCELoss()
        criterion_days = nn.MSELoss()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            codes_output, days_output = self.dependency_model(padded_codes, padded_lengths, padded_days)
            loss_codes = criterion_codes(codes_output, padded_codes[:, 1:, :])
            loss_days = criterion_days(days_output, padded_days[:, 1:])
            loss = loss_codes + loss_days
            loss.backward()
            optimizer.step()

    def generate_admission(self, patient_history):
        codes, lengths, days = self.prepare_sequence(patient_history)
        with torch.no_grad():
            condition = self.dependency_model(codes.unsqueeze(0), lengths.unsqueeze(0), days.unsqueeze(0), export=True)
            predicted_codes = torch.sigmoid(self.dependency_model.proj3(condition[0, -1]))
            predicted_days = torch.relu(self.dependency_model.proj_days(condition[0, -1])).item()
        return predicted_codes.numpy(), max(1, int(predicted_days))

    def forge(self, num_patients):
        synthetic_data = []
        for patient_id in range(num_patients):
            num_admissions = np.random.choice(list(self.admissions_distribution.keys()),
                                              p=[v/sum(self.admissions_distribution.values()) for v in self.admissions_distribution.values()])
            
            patient_data = []
            for admission_id in range(num_admissions):
                if admission_id == 0:
                    new_admission = self.original_forge.forge(1)
                    new_admission['days_from_last_admission'] = 0
                else:
                    predicted_codes, predicted_days = self.generate_admission(pd.DataFrame(patient_data))
                    new_admission = self.original_forge.forge(1)
                    
                    code_cols = [col for col in new_admission.columns if col.startswith('code_')]
                    new_admission[code_cols] = (predicted_codes > 0.5).astype(int)
                    new_admission['days_from_last_admission'] = predicted_days
                
                new_admission['patient_id'] = patient_id
                new_admission['admission_id'] = admission_id
                patient_data.append(new_admission.iloc[0])
            
            synthetic_data.extend(patient_data)
        
        return pd.DataFrame(synthetic_data)

#        


def load_pkl(name):
    with open(name+'.pkl', 'rb') as f:
        data = pickle.load(f)
    return data        



def forge(n, bnds, num_trees, params, class_probs, factor_cols, p, orig_colnames, dist, levels, object_cols):

    """This part is for data generation (FORGE)

    :param n: Number of synthetic samples to generate.
    :type n: int
    :return: Returns generated data.
    :rtype: pandas.DataFrame
    """
    try:
      getattr( 'bnds')
    except AttributeError:
      raise AttributeError('need density estimates to generate data -- run .forde() first!')

    # Sample new observations and get their terminal nodes
    # Draw random leaves with probability proportional to coverage
    unique_bnds = bnds[['tree', 'nodeid', 'cvg']].drop_duplicates()
    draws = np.random.choice(a=range(unique_bnds.shape[0]), p = unique_bnds['cvg'] / num_trees, size=n)
    sampled_trees_nodes = unique_bnds[['tree','nodeid']].iloc[draws,].reset_index(drop =True).reset_index().rename(columns={'index': 'obs'})

    # Get distributions parameters for each new obs.
    if np.invert(factor_cols).any():
      obs_params = pd.merge(sampled_trees_nodes, params, on = ["tree", "nodeid"]).sort_values(by=['obs'], ignore_index = True)
    
    # Get probabilities for each new obs.
    if factor_cols.any():
      obs_probs = pd.merge(sampled_trees_nodes, class_probs, on = ["tree", "nodeid"]).sort_values(by=['obs'], ignore_index = True)
    
    # Sample new data from mixture distribution over trees
    data_new = pd.DataFrame(index=range(n), columns=range(p))
    for j in range(p): 
      colname = orig_colnames[j]
      
      if factor_cols[j]:
        # Factor columns: Multinomial distribution
        data_new.isetitem(j, obs_probs[obs_probs["variable"] == colname].groupby("obs").sample(weights = "prob")["value"].reset_index(drop = True))

      else:
        # Continuous columns: Match estimated distribution parameters with r...() function
        if dist == "truncnorm":
         # sample from normal distribution, only here for debugging
         # data_new.loc[:, j] = np.random.normal(obs_params.loc[obs_params["variable"] == colname, "mean"], obs_params.loc[obs_params["variable"] == colname, "sd"], size = n) 
         
         # sample from truncated normal distribution
         # note: if sd == 0, truncnorm will return location parameter -> this is desired; if we have 
         # all obs. in that leave having the same value, we sample a new obs. with exactly that value as well
         myclip_a = obs_params.loc[obs_params["variable"] == colname, "min"]
         myclip_b = obs_params.loc[obs_params["variable"] == colname, "max"]
         myloc = obs_params.loc[obs_params["variable"] == colname, "mean"]
         myscale = obs_params.loc[obs_params["variable"] == colname, "sd"]
         data_new.isetitem(j, scipy.stats.truncnorm(a =(myclip_a - myloc) / myscale,b = (myclip_b - myloc) / myscale, loc = myloc , scale = myscale ).rvs(size = n))
         del(myclip_a,myclip_b,myloc,myscale)
        else:
          raise ValueError('Other distributions not yet implemented')
    
    # Use original column names
    data_new = data_new.set_axis(orig_colnames, axis = 1, copy = False)
    
    # Convert categories back to category   
    for col in orig_colnames:
      if factor_cols[col]:
        data_new[col] = pd.Categorical.from_codes(data_new[col], categories = levels[col])

    # Convert object columns back to object
    for col in orig_colnames:
      if object_cols[col]:
        data_new[col] = data_new[col].astype("object")

    # Return newly sampled data
    return data_new
# Usage

arf = load_pkl(file_data+folder+"arf_2")
params = arf["params"]

test_ehr_dataset,train_ehr_dataset,synthetic_ehr_dataset,features  = load_create_ehr(read_ehr,save_ehr,file_path_dataset,sample_patients_path,file,valid_perc,features_path,name_file_ehr,type_file='ARFpkl')

config = Config(X)
original_forge = forge()
dependency_model = DependencyModel(config)
enhanced_forge = EnhancedForge(original_forge, dependency_model, config)

# Train the model on real data
enhanced_forge.train_model(train_ehr_dataset,2)

# Generate synthetic data
synthetic_data = enhanced_forge.forge(num_patients=100)