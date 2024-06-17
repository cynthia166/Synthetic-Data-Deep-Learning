
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 


warnings.filterwarnings('ignore') 
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import joblib 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disabling gpu usage because my cuda is corrupted, needs to be fixed. 

import sys
import numpy as np , pandas as pd
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os, warnings, sys
from re import T
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.backend import random_normal
from tensorflow.keras.optimizers import Adam
import generative_model.utils_arf as utils_arf

def get_mnist_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    # mnist_digits = np.concatenate([x_train, x_test], axis=0)
    # mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_digits = x_train.astype("float32") / 255
    return mnist_digits

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BaseVariationalAutoencoder(Model, ABC):
    def __init__(self,  
            seq_len, 
            feat_dim,  
            latent_dim,
            reconstruction_wt = 3.0,
            **kwargs  ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean( name="reconstruction_loss" )
        self.kl_loss_tracker = Mean(name="kl_loss")

        self.encoder = None
        self.decoder = None


    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1: x_decoded = x_decoded.reshape((1, -1))
        return x_decoded


    def get_num_trainable_variables(self):
        trainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.trainable_weights]))
        nonTrainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights]))
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams


    def get_prior_samples(self, num_samples):
        Z = np.random.randn(num_samples, self.latent_dim)
        samples = self.decoder.predict(Z)
        return samples
    

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        return samples


    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    
    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()



    def _get_reconstruction_loss(self, X, X_recons): 

        def get_reconst_loss_by_axis(X, X_c, axis): 
            x_r = tf.reduce_mean(X, axis = axis)
            x_c_r = tf.reduce_mean(X_recons, axis = axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall    
        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)
      
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[2])     # by time axis        
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis
        return reconst_loss
    


    def train_step(self, X):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)

            reconstruction = self.decoder(z)

            reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
            # kl_loss = kl_loss / self.latent_dim

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    
    def test_step(self, X): 
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
        # kl_loss = kl_loss / self.latent_dim

        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    def save_weights(self, model_dir, file_pref): 
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(encoder_wts, os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        joblib.dump(decoder_wts, os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

    
    def load_weights(self, model_dir, file_pref):
        encoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        decoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)


    def save(self, model_dir, file_pref): 

        self.save_weights(model_dir, file_pref)
        dict_params = {

            'seq_len': self.seq_len,
            'feat_dim': self.feat_dim,
            'latent_dim': self.latent_dim,
            'reconstruction_wt': self.reconstruction_wt,
            'hidden_layer_sizes': self.hidden_layer_sizes,
        }
        params_file = os.path.join(model_dir, f'{file_pref}parameters.pkl') 
        joblib.dump(dict_params, params_file)
class TimeVAE(BaseVariationalAutoencoder):    


    def __init__(self,  hidden_layer_sizes, trend_poly = 0, num_gen_seas = 0, custom_seas = None, 
            use_scaler = False, use_residual_conn = True,  **kwargs   ):
        '''
            hidden_layer_sizes: list of number of filters in convolutional layers in encoder and residual connection of decoder. 
            trend_poly: integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term. 
            num_gen_seas: Number of sine-waves to use to model seasonalities. Each sine wae will have its own amplitude, frequency and phase. 
            custom_seas: list of tuples of (num_seasons, len_per_season). 
                num_seasons: number of seasons per cycle. 
                len_per_season: number of epochs (time-steps) per season.
            use_residual_conn: boolean value indicating whether to use a residual connection for reconstruction in addition to
            trend, generic and custom seasonalities.        
        '''

        super(TimeVAE, self).__init__(**kwargs)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.num_gen_seas = num_gen_seas
        self.custom_seas = custom_seas
        self.use_scaler = use_scaler
        self.use_residual_conn = use_residual_conn
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder() 


    def _get_encoder(self):
        encoder_inputs = Input(shape=(self.seq_len, self.feat_dim), name='encoder_input')
        x = encoder_inputs
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                    filters = num_filters, 
                    kernel_size=3, 
                    strides=2, 
                    activation='relu', 
                    padding='same',
                    name=f'enc_conv_{i}')(x)

        x = Flatten(name='enc_flatten')(x)

        # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        self.encoder_last_dense_dim = x.get_shape()[-1]        

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])     
        self.encoder_output = encoder_output
        
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder")
        # encoder.summary()
        return encoder


    def _get_decoder(self):
        decoder_inputs = Input(shape=(int(self.latent_dim)), name='decoder_input')    

        outputs = None
        outputs = self.level_model(decoder_inputs)        

        # trend polynomials
        if self.trend_poly is not None and self.trend_poly > 0: 
            trend_vals = self.trend_model(decoder_inputs)
            outputs = trend_vals if outputs is None else outputs + trend_vals 

        # # generic seasonalities
        # if self.num_gen_seas is not None and self.num_gen_seas > 0:
        #     gen_seas_vals, freq, phase, amplitude = self.generic_seasonal_model(decoder_inputs)
        #     # gen_seas_vals = self.generic_seasonal_model2(decoder_inputs)
        #     outputs = gen_seas_vals if outputs is None else outputs + gen_seas_vals 

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0: 
            cust_seas_vals = self.custom_seasonal_model(decoder_inputs)
            outputs = cust_seas_vals if outputs is None else outputs + cust_seas_vals 


        if self.use_residual_conn:
            residuals = self._get_decoder_residual(decoder_inputs)  
            outputs = residuals if outputs is None else outputs + residuals 


        if self.use_scaler and outputs is not None: 
            scale = self.scale_model(decoder_inputs)
            outputs *= scale

        # outputs = Activation(activation='sigmoid')(outputs)

        if outputs is None: 
            raise Exception('''Error: No decoder model to use. 
            You must use one or more of:
            trend, generic seasonality(ies), custom seasonality(ies), and/or residual connection. ''')
        
        # decoder = Model(decoder_inputs, [outputs, freq, phase, amplitude], name="decoder")
        decoder = Model(decoder_inputs, [outputs], name="decoder")
        return decoder


    def level_model(self, z): 
        level_params = Dense(self.feat_dim, name="level_params", activation='relu')(z)
        level_params = Dense(self.feat_dim, name="level_params2")(level_params)
        level_params = Reshape(target_shape=(1, self.feat_dim))(level_params)      # shape: (N, 1, D)

        ones_tensor = tf.ones(shape=[1, self.seq_len, 1], dtype=tf.float32)   # shape: (1, T, D)

        level_vals = level_params * ones_tensor
        # print('level_vals', tf.shape(level_vals))
        return level_vals



    def scale_model(self, z): 
        scale_params = Dense(self.feat_dim, name="scale_params", activation='relu')(z)
        scale_params = Dense(self.feat_dim, name="scale_params2")(scale_params)
        scale_params = Reshape(target_shape=(1, self.feat_dim))(scale_params)      # shape: (N, 1, D)

        scale_vals = tf.repeat(scale_params, repeats = self.seq_len, axis = 1)      # shape: (N, T, D)
        # print('scale_vals', tf.shape(scale_vals))
        return scale_vals




    def trend_model(self, z):
        trend_params = Dense(self.feat_dim * self.trend_poly, name="trend_params", activation='relu')(z)
        trend_params = Dense(self.feat_dim * self.trend_poly, name="trend_params2")(trend_params)
        trend_params = Reshape(target_shape=(self.feat_dim, self.trend_poly))(trend_params)  #shape: N x D x P
        # print("trend params shape", trend_params.shape)
        # shape of trend_params: (N, D, P)  P = num_poly

        lin_space = K.arange(0, float(self.seq_len), 1) / self.seq_len # shape of lin_space : 1d tensor of length T
        poly_space = K.stack([lin_space ** float(p+1) for p in range(self.trend_poly)], axis=0)  # shape: P x T
        # print('poly_space', poly_space.shape, poly_space[0])

        trend_vals = K.dot(trend_params, poly_space)            # shape (N, D, T)
        trend_vals = tf.transpose(trend_vals, perm=[0,2,1])     # shape: (N, T, D)
        trend_vals = K.cast(trend_vals, tf.float32)
        # print('trend_vals shape', tf.shape(trend_vals)) 
        return trend_vals



    def custom_seasonal_model(self, z):

        N = tf.shape(z)[0]
        ones_tensor = tf.ones(shape=[N, self.feat_dim, self.seq_len], dtype=tf.int32)
        
        all_seas_vals = []
        for i, season_tup in enumerate(self.custom_seas):  
            num_seasons, len_per_season = season_tup

            season_params = Dense(self.feat_dim * num_seasons, name=f"season_params_{i}")(z)    # shape: (N, D * S)  
            season_params = Reshape(target_shape=(self.feat_dim, num_seasons))(season_params)  # shape: (N, D, S)  
            # print('\nshape of season_params', tf.shape(season_params))  

            season_indexes_over_time = self._get_season_indexes_over_seq(num_seasons, len_per_season) #shape: (T, )
            # print("season_indexes_over_time shape: ", tf.shape(season_indexes_over_time))

            dim2_idxes = ones_tensor * tf.reshape(season_indexes_over_time, shape=(1,1,-1))         #shape: (1, 1, T)
            # print("dim2_idxes shape: ", tf.shape(dim2_idxes))

            season_vals = tf.gather(season_params, dim2_idxes, batch_dims = -1)                 #shape (N, D, T)
            # print("season_vals shape: ", tf.shape(season_vals))

            all_seas_vals.append(season_vals)
        
        all_seas_vals = K.stack(all_seas_vals, axis=-1)                # shape: (N, D, T, S)
        all_seas_vals = tf.reduce_sum(all_seas_vals, axis=-1)          # shape (N, D, T)
        all_seas_vals = tf.transpose(all_seas_vals, perm=[0,2,1])      # shape (N, T, D)
        # print('final shape:', tf.shape(all_seas_vals))
        return all_seas_vals



    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        curr_len = 0
        season_idx = []
        curr_idx = 0
        while curr_len < self.seq_len:
            reps = len_per_season if curr_len + len_per_season <= self.seq_len else self.seq_len - curr_len
            season_idx.extend([curr_idx] * reps)
            curr_idx += 1
            if curr_idx == num_seasons: curr_idx = 0
            curr_len += reps
        return season_idx

    

    def generic_seasonal_model(self, z):

        freq = Dense(self.feat_dim * self.num_gen_seas, name="g_season_freq", activation='sigmoid')(z)
        freq = Reshape(target_shape=(1, self.feat_dim, self.num_gen_seas))(freq)  # shape: (N, 1, D, S)  

        phase = Dense(self.feat_dim * self.num_gen_seas, name="g_season_phase")(z)
        phase = Reshape(target_shape=(1, self.feat_dim, self.num_gen_seas))(phase)  # shape: (N, 1, D, S)  

        amplitude = Dense(self.feat_dim * self.num_gen_seas, name="g_season_amplitude")(z)
        amplitude = Reshape(target_shape=(1, self.feat_dim, self.num_gen_seas))(amplitude)  # shape: (N, 1, D, S)  

        lin_space = K.arange(0, float(self.seq_len), 1) / self.seq_len # shape of lin_space : 1d tensor of length T
        lin_space = tf.reshape(lin_space, shape=(1, self.seq_len, 1, 1))      #shape: 1, T, 1, 1 
        # print('lin_space:', lin_space)      

        seas_vals = amplitude * K.sin( 2. * np.pi * freq * lin_space + phase )        # shape: N, T, D, S
        seas_vals = tf.math.reduce_sum(seas_vals, axis = -1)                    # shape: N, T, D

        # print('seas_vals:', seas_vals)      
        return seas_vals



    def generic_seasonal_model2(self, z):

        season_params = Dense(self.feat_dim * self.num_gen_seas, name="g_season_params")(z)
        season_params = Reshape(target_shape=(self.feat_dim, self.num_gen_seas))(season_params)  # shape: (D, S)  

        p = self.num_gen_seas
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)

        ls = K.arange(0, float(self.seq_len), 1) / self.seq_len # shape of ls : 1d tensor of length T

        s1 = K.stack([K.cos(2 * np.pi * i * ls) for i in range(p1)], axis=0)
        s2 = K.stack([K.sin(2 * np.pi * i * ls) for i in range(p2)], axis=0)
        if p == 1:
            s = s2
        else:
            s = K.concatenate([s1, s2], axis=0)
        s = K.cast(s, np.float32)   

        seas_vals = K.dot(season_params, s, name='g_seasonal_vals')
        seas_vals = tf.transpose(seas_vals, perm=[0,2,1])     # shape: (N, T, D)
        seas_vals = K.cast(seas_vals, np.float32)
        print('seas_vals shape', tf.shape(seas_vals)) 

        return seas_vals



    def _get_decoder_residual(self, x):

        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation='relu')(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(x)

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters = num_filters, 
                    kernel_size=3, 
                    strides=2, 
                    padding='same',
                    activation='relu', 
                    name=f'dec_deconv_{i}')(x)

        # last de-convolution
        x = Conv1DTranspose(
                filters = self.feat_dim, 
                    kernel_size=3, 
                    strides=2, 
                    padding='same',
                    activation='relu', 
                    name=f'dec_deconv__{i+1}')(x)

        x = Flatten(name='dec_flatten')(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        residuals = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        return residuals


    def save(self, model_dir, file_pref): 

        super().save_weights(model_dir, file_pref)
        dict_params = {
            'seq_len': self.seq_len,
            'feat_dim': self.feat_dim,
            'latent_dim': self.latent_dim,
            'reconstruction_wt': self.reconstruction_wt,

            'hidden_layer_sizes': self.hidden_layer_sizes,
            'trend_poly': self.trend_poly,
            'num_gen_seas': self.num_gen_seas,
            'custom_seas': self.custom_seas,
            'use_scaler': self.use_scaler,
            'use_residual_conn': self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f'{file_pref}parameters.pkl') 
        joblib.dump(dict_params, params_file)


    @staticmethod
    def load(model_dir, file_pref):
        params_file = os.path.join(model_dir, f'{file_pref}parameters.pkl') 
        dict_params = joblib.load(params_file)

        vae_model = TimeVAE( **dict_params )

        vae_model.load_weights(model_dir, file_pref)
        
        vae_model.compile(optimizer=Adam())

        return vae_model 
class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self
        

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data

    
class MinMaxScaler_Feat_Dim():
    '''Scales history and forecast parts of time-series based on history data'''
    def __init__(self, scaling_len, input_dim, upper_bound = 3., lower_bound = -3.):         
        self.scaling_len = scaling_len
        self.min_vals_per_d = None      
        self.max_vals_per_d = None  
        self.input_dim = input_dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        

    def fit(self, X, y=None): 

        if self.scaling_len < 1: 
            msg = f''' Error scaling series. 
            scaling_len needs to be at least 2. Given length is {self.scaling_len}.  '''
            raise Exception(msg)

        X_f = X[ :,  : self.scaling_len , : ]
        self.min_vals_per_d = np.expand_dims(np.expand_dims(X_f.min(axis=0).min(axis=0), axis=0), axis=0)
        self.max_vals_per_d = np.expand_dims(np.expand_dims(X_f.max(axis=0).max(axis=0), axis=0), axis=0)

        self.range_per_d = self.max_vals_per_d - self.min_vals_per_d
        self.range_per_d = np.where(self.range_per_d == 0, 1e-5, self.range_per_d)

        # print(self.min_vals_per_d.shape); print(self.max_vals_per_d.shape)
              
        return self
    
    def transform(self, X, y=None): 
        assert X.shape[-1] == self.min_vals_per_d.shape[-1], "Error: Dimension of array to scale doesn't match fitted array."
         
        X = X - self.min_vals_per_d
        X = np.divide(X, self.range_per_d )        
        X = np.where( X < self.upper_bound, X, self.upper_bound)
        X = np.where( X > self.lower_bound, X, self.lower_bound)
        return X
    
    def fit_transform(self, X, y=None):
        X = X.copy()
        self.fit(X)
        return self.transform(X)
        

    def inverse_transform(self, X):
        X = X.copy()
        X = X * self.range_per_d 
        X = X + self.min_vals_per_d
        # print(X.shape)
        return X
    
def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)


def concat_attributes(a,b):

    # Redimensiona 'b' para que tenga la misma forma que 'a' en los dos primeros ejes
    #b is attributes
    #a = np.random.rand(100, 244, 5)
    #b = np.random.rand(100, 2)

    b = np.repeat(b[:, :, np.newaxis], a.shape[2], axis=2)
    print(b.shape)
    print(a.shape)
    # Ahora 'b' es un array de forma (100, 244, 2)
    print(b.shape)  # Impri   me: (100, 244, 2)

    # Concatena 'a' y 'b' a lo largo del tercer eje
    c = np.concatenate((a, b), axis=1)

# Ahora 'c' es un array de forma (100, 244, 7)
    print(c.shape) 
    return c    

def draw_orig_and_post_pred_sample(orig, reconst, n):

    fig, axs = plt.subplots(n, 2, figsize=(10,6))
    i = 1
    for _ in range(n):
        rnd_idx = np.random.choice(len(orig))
        o = orig[rnd_idx]
        r = reconst[rnd_idx]

        plt.subplot(n, 2, i)
        plt.imshow(o, 
            # cmap='gray', 
            aspect='auto')
        # plt.title("Original")
        i += 1

        plt.subplot(n, 2, i)
        plt.imshow(r, 
            # cmap='gray', 
            aspect='auto')
        # plt.title("Sampled")
        i += 1

    fig.suptitle("Original vs Reconstructed Data", fontsize = 12)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    start = time.time()

    # choose model
    vae_type = 'timeVAE'           # vae_dense, vae_conv, timeVAE
    # ----------------------------------------------------------------------------------
    # read data   
    import pickle
    import gzip
    import os
    ruta ='train_sp/non_prepo/generated/'
    if os.path.exists(ruta):
        print("La ruta existe.")
    else:
        print("La ruta no existe.")
        
    #dataset_name = '/non_prepo/DATASET_NAME_non_prepro'
    dataset_name = '/train_splitDATASET_NAME_prepro'
    features_train = load_data('train_sp' + dataset_name + 'train_data_features.pkl')
    attribute_train = load_data('train_sp'  + dataset_name + 'train_data_attributes.pkl')

    full_train_data = concat_attributes(features_train, attribute_train)
    
    full_train_data = full_train_data
    full_train_data = np.transpose(full_train_data, (0, 2, 1))
    N, T, D = full_train_data.shape  
    full_train_data = full_train_data[:round(N*0.5)] 
    print('data shape:', N, T, D) 

    # ----------------------------------------------------------------------------------
    # further split the training data into train and validation set - same thing done in forecasting task
    valid_perc = 0.25
    N_train = int(N * (1 - valid_perc))
    N_valid = N - N_train

    # Shuffle data
    np.random.shuffle(full_train_data)

    train_data = full_train_data[:N_train]
    valid_data = full_train_data[N_train:]   
    print("train/valid shapes: ", train_data.shape, valid_data.shape)    
    
    # ----------------------------------------------------------------------------------
    # min max scale the data    
    scaler = MinMaxScaler()        
    scaled_train_data = scaler.fit_transform(train_data)

    scaled_valid_data = scaler.transform(valid_data)
    # joblib.dump(scaler, 'scaler.save')  

    # ----------------------------------------------------------------------------------
    # instantiate the model     
    
    latent_dim = 2

    #vae_type == 'timeVAE':
    vae = TimeVAE( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[50, 100, 200],        #[80, 200, 250] 
                reconstruction_wt = 3.0,
                # ---------------------
                # disable following three arguments to use the model as TimeVAE_Base. Enabling will convert to Interpretable version.
                # Also set use_residual_conn= False if you want to only have interpretable components, and no residual (non-interpretable) component. 
                
                # trend_poly=2, 
                # custom_seas = [ (6,1), (7, 1), (8,1), (9,1)] ,     # list of tuples of (num_of_seasons, len_per_season)
                # use_scaler = True,
                
                #---------------------------
                use_residual_conn = True
            )   
    

    
    vae.compile(optimizer=Adam())
    # vae.summary() ; sys.exit()

    early_stop_loss = 'loss'
    early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)

    vae.fit(
        scaled_train_data, 
        batch_size = 16,
        epochs=10,
        shuffle = True,
        callbacks=[early_stop_callback, reduceLR],
        verbose = 1
    )
    
    # ----------------------------------------------------------------------------------    
    # save model 
    model_dir = 'train_sp/non_prepo/generated/'
    vae_type = 'TimeVaeT'
    perc_of_train_used = 1- valid_perc
    file_pref = f'vae_{vae_type}_{perc_of_train_used}_'
    vae.save(model_dir, file_pref)
    
    # ----------------------------------------------------------------------------------
    # visually check reconstruction 
    X = scaled_train_data

    x_decoded = vae.predict(scaled_train_data)
    print('x_decoded.shape', x_decoded.shape)

    ### compare original and posterior predictive (reconstructed) samples
    draw_orig_and_post_pred_sample(X, x_decoded, n=5)
    

    # # Plot the prior generated samples over different areas of the latent space
    #if latent_dim == 2: utils.plot_latent_space_timeseries(vae, n=8, figsize = (20, 10))
        
    # # ----------------------------------------------------------------------------------
    # draw random prior samples
    num_samples = N
    print("num_samples: ", num_samples)

    samples = vae.get_prior_samples(num_samples=num_samples)
    
    #utils.plot_samples(samples, n=5)

    # inverse-transform scaling 
    samples = scaler.inverse_transform(samples)
    print('shape of gen samples: ', samples.shape) 

    # ----------------------------------------------------------------------------------
    # save samples
    output_dir = 'train_sp/non_prepo/generated/'
    sample_fname = f'{vae_type}_gen_samples2.npz' 
    samples_fpath = os.path.join(output_dir, sample_fname) 
    np.savez_compressed(samples_fpath, data=samples)

    # ----------------------------------------------------------------------------------
    
    # later.... load model 
    #new_vae = TimeVAE.load(model_dir, file_pref)

    #new_x_decoded = new_vae.predict(scaled_train_data)
    # print('new_x_decoded.shape', new_x_decoded.shape)

    #print('Preds from orig and loaded models equal: ', np.allclose( x_decoded,  new_x_decoded, atol=1e-5))        
    
    # ----------------------------------------------------------------------------------
    