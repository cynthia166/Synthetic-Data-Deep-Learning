import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import pickle



def save_pkl(data,name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(data, f)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_admissions(real_data, synthetic_data, feature_columns):
    real_data = real_data.reset_index(drop=True)
    synthetic_data = synthetic_data.reset_index(drop=True)
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Number of feature columns: {len(feature_columns)}")

    # Normalize features
    scaler = StandardScaler()
    real_features = scaler.fit_transform(real_data[feature_columns])
    synthetic_features = scaler.transform(synthetic_data[feature_columns])

    print(f"Real features shape: {real_features.shape}")
    print(f"Synthetic features shape: {synthetic_features.shape}")

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(real_features, synthetic_features)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Sort patients by number of admissions (descending)
    patient_admission_counts = real_data['id_patient'].value_counts().sort_values(ascending=False)
    
    matched_synthetic_data = []
    used_synthetic_indices = set()

    for patient_id in patient_admission_counts.index:
        patient_admissions = real_data[real_data['id_patient'] == patient_id].sort_values('visit_rank')
        
        for _, real_admission in patient_admissions.iterrows():
            real_index = real_admission.name
            similarity_scores = similarity_matrix[real_index]
            
            # Find the best match that hasn't been used yet
            for synthetic_index in np.argsort(similarity_scores)[::-1]:
                if synthetic_index not in used_synthetic_indices:
                    best_match_index = synthetic_index
                    used_synthetic_indices.add(best_match_index)
                    break
            else:
                print(f"Warning: No unused synthetic admission found for patient {patient_id}, visit {real_admission['visit_rank']}")
                continue

            synthetic_admission = synthetic_data.iloc[best_match_index].copy()
            synthetic_admission['id_patient'] = real_admission['id_patient']
            synthetic_admission['visit_rank'] = real_admission['visit_rank']
            synthetic_admission['similarity_score'] = similarity_scores[best_match_index]
            matched_synthetic_data.append(synthetic_admission)

    matched_synthetic_df = pd.DataFrame(matched_synthetic_data)
    
    return matched_synthetic_df

def find_similar_admissions_(real_data, synthetic_data, feature_columns):
    real_data = real_data.reset_index(drop = True)
    synthetic_data = synthetic_data.reset_index(drop = True)
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Number of feature columns: {len(feature_columns)}")

    # Normalize features
    scaler = StandardScaler()
     
    real_features = scaler.fit_transform(real_data[feature_columns])
   
    synthetic_features = scaler.transform(synthetic_data[feature_columns])

    print(f"Real features shape: {real_features.shape}")
    print(f"Synthetic features shape: {synthetic_features.shape}")

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(real_features, synthetic_features)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Find the most similar synthetic admission for each real admission
    matched_synthetic_data = []
    for i, real_admission in real_data.iterrows():
        print(i)
        best_match_index = np.argmax(similarity_matrix[i])
        synthetic_admission = synthetic_data.iloc[best_match_index].copy()
        synthetic_admission['id_patient'] = real_admission['id_patient']
        synthetic_admission['visit_rank'] = real_admission['visit_rank']
        synthetic_admission['similarity_score'] = similarity_matrix[i, best_match_index]
        matched_synthetic_data.append(synthetic_admission)

    matched_synthetic_df = pd.DataFrame(matched_synthetic_data)
    
    return matched_synthetic_df