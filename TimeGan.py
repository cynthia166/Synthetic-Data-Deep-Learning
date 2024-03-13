"""
Reference: "Time-series Generative Adversarial Networks", Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar
"""
# stdlib
from pathlib import Path
from preprocess_input_SDmodel import *
from typing import Any, Callable, List, Optional, Tuple, Union
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader

# third party
import numpy as np
import pandas as pd

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_encoder import BinEncoder, TabularEncoder
from synthcity.plugins.core.models.ts_model import TimeSeriesModel, modes
from synthcity.plugins.core.models.ts_tabular_gan import TimeSeriesTabularGAN
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE
from synthcity.utils.samplers import ImbalancedDatasetSampler


class TimeGANPlugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.time_series.plugin_timegan.TimeGANPlugin
        :parts: 1


    Synthetic time series generation using TimeGAN.

    Args:
        n_iter: int
            Maximum number of iterations in the Generator.
        n_units_in: int
            Number of features
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_nonlin: string, default 'elu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_batch_norm: bool
            Enable/disable batch norm for the generator
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        generator_residual: bool
            Use residuals for the generator
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_nonlin: string, default 'relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
        discriminator_batch_norm: bool
            Enable/disable batch norm for the discriminator
        discriminator_dropout: float
            Dropout value for the discriminator. If 0, the dropout is not used.
        lr: float
            learning rate for optimizer.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        random_state: int
            random_state used
        clipping_value: int, default 0
            Gradients clipping value
        gamma_penalty
            Latent representation penalty
        moments_penalty: float = 100
            Moments(var and mean) penalty
        embedding_penalty: float = 10
            Embedding representation penalty
        mode: str = "RNN"
            Core neural net architecture.
            Available models:
                - "LSTM"
                - "GRU"
                - "RNN"
                - "Transformer"
                - "MLSTM_FCN"
                - "TCN"
                - "InceptionTime"
                - "InceptionTimePlus"
                - "XceptionTime"
                - "ResCNN"
                - "OmniScaleCNN"
                - "XCM"
        device
            The device used by PyTorch. cpu/cuda
        use_horizon_condition: bool. Default = True
            Whether to condition the covariate generation on the observation times or not.
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        encoder:
            Pre-trained tabular encoder. If None, a new encoder is trained.
        # Core Plugin arguments
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.

    Example:
        >>> from synthcity.plugins import Plugins
        >>> from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
        >>> from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        >>> static, temporal, horizons, outcome = GoogleStocksDataloader().load()
        >>> loader = TimeSeriesDataLoader(
        >>>             temporal_data=temporal,
        >>>             observation_times=horizons,
        >>>             static_data=static,
        >>>             outcome=outcome,
        >>> )
        >>>
        >>> plugin = Plugins().get("timegan", n_iter = 50)
        >>> plugin.fit(loader)
        >>>
        >>> plugin.generate(count = 10)

    """

    def __init__(
        self,
        n_iter: int = 1000,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 150,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out_discrete: str = "softmax",
        generator_nonlin_out_continuous: str = "tanh",
        generator_batch_norm: bool = False,
        generator_dropout: float = 0.01,
        generator_loss: Optional[Callable] = None,
        generator_lr: float = 1e-3,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_loss: Optional[Callable] = None,
        discriminator_lr: float = 1e-3,
        discriminator_weight_decay: float = 1e-3,
        batch_size: int = 1,
        n_iter_print: int = 10,
        clipping_value: int = 0,
        encoder_max_clusters: int = 20,
        encoder: Any = None,
        device: Any = DEVICE,
        mode: str = "RNN",
        gamma_penalty: float = 1,
        moments_penalty: float = 100,
        embedding_penalty: float = 10,
        use_horizon_condition: bool = True,
        dataloader_sampling_strategy: str = "imbalanced_time_censoring",  # none, imbalanced_censoring, imbalanced_time_censoring
        # core plugin arguments
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
        )

        log.info(
            f"""TimeGAN: mode = {mode} dataloader_sampling_strategy = {dataloader_sampling_strategy}"""
        )
        self.n_iter = n_iter
        self.generator_n_layers_hidden = generator_n_layers_hidden
        self.generator_n_units_hidden = generator_n_units_hidden
        self.generator_nonlin = generator_nonlin
        self.generator_nonlin_out_discrete = generator_nonlin_out_discrete
        self.generator_nonlin_out_continuous = generator_nonlin_out_continuous
        self.generator_batch_norm = generator_batch_norm
        self.generator_dropout = generator_dropout
        self.generator_loss = generator_loss
        self.generator_lr = generator_lr
        self.generator_weight_decay = generator_weight_decay
        self.generator_residual = generator_residual
        self.discriminator_n_layers_hidden = discriminator_n_layers_hidden
        self.discriminator_n_units_hidden = discriminator_n_units_hidden
        self.discriminator_nonlin = discriminator_nonlin
        self.discriminator_n_iter = discriminator_n_iter
        self.discriminator_batch_norm = discriminator_batch_norm
        self.discriminator_dropout = discriminator_dropout
        self.discriminator_loss = discriminator_loss
        self.discriminator_lr = discriminator_lr
        self.discriminator_weight_decay = discriminator_weight_decay
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.mode = mode
        self.encoder_max_clusters = encoder_max_clusters
        self.encoder = encoder
        self.device = device
        self.gamma_penalty = gamma_penalty
        self.moments_penalty = moments_penalty
        self.embedding_penalty = embedding_penalty
        self.use_horizon_condition = use_horizon_condition
        self.dataloader_sampling_strategy = dataloader_sampling_strategy

        self.outcome_encoder = TabularEncoder(max_clusters=encoder_max_clusters)

    @staticmethod
    def name() -> str:
        return "timegan"

    @staticmethod
    def type() -> str:
        return "time_series"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_iter", low=100, high=1000, step=100),
            IntegerDistribution(name="generator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="generator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="generator_nonlin", choices=["relu", "leaky_relu", "tanh", "elu"]
            ),
            FloatDistribution(name="generator_dropout", low=0, high=0.2),
            IntegerDistribution(name="discriminator_n_layers_hidden", low=1, high=4),
            IntegerDistribution(
                name="discriminator_n_units_hidden", low=50, high=150, step=50
            ),
            CategoricalDistribution(
                name="discriminator_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            IntegerDistribution(name="discriminator_n_iter", low=1, high=5),
            FloatDistribution(name="discriminator_dropout", low=0, high=0.2),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[50, 100, 250]),
            IntegerDistribution(name="encoder_max_clusters", low=2, high=20),
            CategoricalDistribution(name="mode", choices=modes),
            FloatDistribution(name="gamma_penalty", low=0, high=1000),
            FloatDistribution(name="moments_penalty", low=0, high=1000),
            FloatDistribution(name="embedding_penalty", low=0, high=1000),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "TimeGANPlugin":
        if X.type() not in ["time_series", "time_series_survival"]:
            raise ValueError(f"Invalid data type = {X.type()}")

        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        sampler: Optional[ImbalancedDatasetSampler] = None

        if "cond" in kwargs:
            cond = kwargs["cond"]

        # Static and temporal generation
        if X.type() == "time_series":
            static, temporal, observation_times, outcome = X.unpack(pad=True)
        elif X.type() == "time_series_survival":
            static, temporal, observation_times, T, E = X.unpack(pad=True)
            outcome = pd.concat([pd.Series(T), pd.Series(E)], axis=1)
            outcome.columns = ["time_to_event", "event"]

            sampling_labels: Optional[list] = None

            if self.dataloader_sampling_strategy == "imbalanced_censoring":
                sampling_labels = list(E.values)
            elif self.dataloader_sampling_strategy == "imbalanced_time_censoring":
                Tbins = (
                    BinEncoder().fit_transform(T.to_frame()).values.squeeze().tolist()
                )
                sampling_labels = list(zip(E, Tbins))

            if sampling_labels is not None:
                sampler = ImbalancedDatasetSampler(sampling_labels)

        self.cov_model = TimeSeriesTabularGAN(
            static_data=static,
            temporal_data=temporal,
            observation_times=observation_times,
            cond=cond,
            generator_n_iter=self.n_iter,
            generator_n_layers_hidden=self.generator_n_layers_hidden,
            generator_n_units_hidden=self.generator_n_units_hidden,
            generator_nonlin=self.generator_nonlin,
            generator_nonlin_out_discrete=self.generator_nonlin_out_discrete,
            generator_nonlin_out_continuous=self.generator_nonlin_out_continuous,
            generator_batch_norm=self.generator_batch_norm,
            generator_dropout=self.generator_dropout,
            generator_loss=self.generator_loss,
            generator_lr=self.generator_lr,
            generator_weight_decay=self.generator_weight_decay,
            generator_residual=self.generator_residual,
            discriminator_n_layers_hidden=self.discriminator_n_layers_hidden,
            discriminator_n_units_hidden=self.discriminator_n_units_hidden,
            discriminator_nonlin=self.discriminator_nonlin,
            discriminator_n_iter=self.discriminator_n_iter,
            discriminator_batch_norm=self.discriminator_batch_norm,
            discriminator_dropout=self.discriminator_dropout,
            discriminator_loss=self.discriminator_loss,
            discriminator_lr=self.discriminator_lr,
            discriminator_weight_decay=self.discriminator_weight_decay,
            batch_size=self.batch_size,
            n_iter_print=self.n_iter_print,
            random_state=self.random_state,
            clipping_value=self.clipping_value,
            mode=self.mode,
            encoder_max_clusters=self.encoder_max_clusters,
            encoder=self.encoder,
            device=self.device,
            gamma_penalty=self.gamma_penalty,
            moments_penalty=self.moments_penalty,
            embedding_penalty=self.embedding_penalty,
            use_horizon_condition=self.use_horizon_condition,
            dataloader_sampler=sampler,
        )
        self.cov_model.fit(static, temporal, observation_times, cond=cond)

        # Outcome generation
        self.outcome_encoder.fit(outcome)
        outcome_enc = self.outcome_encoder.transform(outcome)
        self.outcome_encoded_columns = outcome_enc.columns

        self.outcome_model = TimeSeriesModel(
            task_type="regression",
            n_static_units_in=static.shape[-1],
            n_temporal_units_in=temporal[0].shape[-1],
            n_temporal_window=temporal[0].shape[0],
            output_shape=outcome_enc.shape[1:],
            n_static_units_hidden=self.generator_n_units_hidden,
            n_static_layers_hidden=self.generator_n_layers_hidden,
            n_temporal_units_hidden=self.generator_n_units_hidden,
            n_temporal_layers_hidden=self.generator_n_layers_hidden,
            n_iter=self.n_iter,
            mode=self.mode,
            n_iter_print=self.n_iter_print,
            batch_size=self.batch_size,
            lr=self.generator_lr,
            weight_decay=self.generator_weight_decay,
            window_size=1,
            device=self.device,
            dropout=self.generator_dropout,
            nonlin=self.generator_nonlin,
            nonlin_out=self.outcome_encoder.activation_layout(
                discrete_activation="softmax",
                continuous_activation="tanh",
            ),
        )
        self.outcome_model.fit(
            np.asarray(static),
            np.asarray(temporal),
            np.asarray(observation_times),
            np.asarray(outcome_enc),
        )

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        cond: Optional[Union[pd.DataFrame, pd.Series]] = None
        static_data_cond: Optional[pd.DataFrame] = None
        observation_times_cond: Optional[list] = None

        if "cond" in kwargs:
            cond = kwargs["cond"]
        if "static_data" in kwargs:
            static_data_cond = kwargs["static_data"]
        if "observation_times" in kwargs:
            observation_times_cond = kwargs["observation_times"]

        def _sample(count: int) -> Tuple:
            local_cond: Optional[Union[pd.DataFrame, pd.Series]] = None
            local_static_data: Optional[pd.DataFrame] = None
            local_observation_times: Optional[list] = None
            if cond is not None:
                local_cond = cond.sample(count, replace=True)
            if static_data_cond is not None:
                local_static_data = static_data_cond.sample(count, replace=True)
            if observation_times_cond is not None:
                ids = list(range(len(observation_times_cond)))
                local_ids = np.random.choice(ids, count, replace=True)
                local_observation_times = np.asarray(observation_times_cond)[
                    local_ids
                ].tolist()

            static, temporal, observation_times = self.cov_model.generate(
                count,
                cond=local_cond,
                static_data=local_static_data,
                observation_times=local_observation_times,
            )

            outcome_enc = pd.DataFrame(
                self.outcome_model.predict(
                    np.asarray(static),
                    np.asarray(temporal),
                    np.asarray(observation_times),
                ),
                columns=self.outcome_encoded_columns,
            )
            outcome_raw = self.outcome_encoder.inverse_transform(outcome_enc)
            outcome = pd.DataFrame(
                outcome_raw, columns=self.data_info["outcome_features"]
            )

            if self.data_info["data_type"] == "time_series":
                return static, temporal, observation_times, outcome
            elif self.data_info["data_type"] == "time_series_survival":
                return (
                    static,
                    temporal,
                    observation_times,
                    outcome[self.data_info["time_to_event_column"]],
                    outcome[self.data_info["event_column"]],
                )
            else:
                raise RuntimeError("unknow data type")

        return self._safe_generate_time_series(_sample, count, syn_schema)
'''
plugin = TimeGANPlugin

type_df = "entire"
arhivo = 'aux/raw_input.csv'
name  = "input_generative_g.csv"

#res = obtener_added_cols_targer_visitrank_(arhivo,name)
cols_to_drop1 = ['ADMITTIME','HADM_ID']



res = pd.read_csv("aux/full_input.csv")
keywords = ['INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY','GENDER','SUBJECT']
 
def get_input_( type_df,arhivo,name,cols_to_drop1,res,keywords):
# Obtener una lista de pacientes únicos
    unique_patients = res['SUBJECT_ID'].unique()

    # Calcular el 20% del total de pacientes únicos
    sample_size = int(0.03 * len(unique_patients))
    # Obtener una muestra aleatoria del 20% de los pacientes únicos
    sample_patients = np.random.choice(unique_patients, size=sample_size, replace=False)
    # Filtrar el DataFrame para incluir solo los registros de los pacientes en la muestra
    sample_df = res[res['SUBJECT_ID'].isin(sample_patients)]

    dfs = obtener_entire(sample_df,type_df)

    # Lista de nombres de columnas a buscar
    # Supongamos que type_df es tu DataFrame
      # Filtrar las columnas de type_df que contienen las palabras clave
    static_data_cols = [col for col in dfs.columns if any(keyword in col for keyword in keywords)]
    # Crear un nuevo DataFrame que solo incluye las columnas filtradas
    not_considet_temporal = [
    i for i in static_data_cols if i != 'SUBJECT_ID'
    ]

    static_data = dfs[static_data_cols].groupby('SUBJECT_ID').max().reset_index(drop=True)
    outcomes = dfs[['HOSPITAL_EXPIRE_FLAG','SUBJECT_ID']].groupby('SUBJECT_ID').max().reset_index(drop=True)
    temporal_data = dfs.drop(columns=not_considet_temporal+['HOSPITAL_EXPIRE_FLAG'])
    cols_to_drop = list(temporal_data.filter(like='Unnamed', axis=1).columns) + ['ADMITTIME','HADM_ID']
    temporal_data.drop(cols_to_drop, axis=1, inplace=True)
    
   



    # Establecer 'SUBJECT_ID' y 'visit_rank' como índices
    temporal_data.set_index(['SUBJECT_ID', 'visit_rank'], inplace=True)
    # Ordenar temporal_data por el índice 'visit_rank' en orden ascendente
    temporal_data.sort_index(level='visit_rank', inplace=True)
    # Ahora se agrupa por 'SUBJECT_ID' y se recoge la información necesaria.
    grouped = temporal_data.groupby(level='SUBJECT_ID')

    # Se obtiene 'observations' como los índices de segundo nivel para cada grupo.
    observation_data = [group.index.get_level_values('visit_rank').tolist() for _, group in grouped]

    # Se obtiene los dataframes temporales por grupo, reseteando el índice 'SUBJECT_ID'.
    temporal_dataframes = [group.reset_index(level=0, drop=True) for _, group in grouped]
    return static_data, temporal_dataframes, observation_data,outcomes

static_data, temporal_dataframes, observation_data,outcomes = get_input_( type_df,arhivo,name,cols_to_drop1,res,keywords)

loader = TimeSeriesDataLoader(
    temporal_data=temporal_dataframes,
    observation_times=observation_data,
    static_data=static_data,
    outcome=outcomes,
model.fit(loader))'''

model = TimeGANPlugin()



from synthcity.plugins import Plugins
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
static, temporal, horizons, outcome = GoogleStocksDataloader().load()
for i in range(temporal):

    i.fillna(i.mean(), inplace=True)

# O reemplazar los NaNs con la media de la columna
        
loader = TimeSeriesDataLoader(
            temporal_data=temporal,
            observation_times=horizons,
            static_data=static,
            outcome=outcome,
)
    
model.fit(loader)
model.generate(count = 10)