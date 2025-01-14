import os
import time
import json
import warnings
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization, Attention, Input
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import keras_tuner as kt

warnings.filterwarnings("ignore")  # Ocultar warnings do Numpy

# Carregar configurações externas
CONFIGS_PATH = os.path.join(os.path.dirname(__file__), 'configs.json')
configs = json.loads(open(CONFIGS_PATH).read())

# Função para construir um modelo aprimorado de LSTM com atenção
def build_network(hp):
    layers = [
        hp.Int('time_steps', min_value=10, max_value=100, step=10),
        hp.Int('features', min_value=1, max_value=10, step=1),
        hp.Int('lstm_1_units', min_value=32, max_value=256, step=32),
        hp.Int('lstm_2_units', min_value=32, max_value=256, step=32),
        hp.Int('lstm_3_units', min_value=16, max_value=128, step=16),
        hp.Int('output_dim', min_value=1, max_value=10, step=1)
    ]
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    inputs = Input(shape=(layers[0], layers[1]))  # Entrada com dimensão (time_steps, features)

    # Primeira camada LSTM com retorno de sequência
    x = LSTM(layers[2], return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # Segunda camada LSTM com retorno de sequência
    x = LSTM(layers[3], return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # Mecanismo de atenção
    attention = Attention()([x, x])

    # Última camada LSTM com retorno de saída única
    x = LSTM(layers[4], return_sequences=False)(attention)
    x = Dropout(0.2)(x)

    # Camada densa final
    outputs = Dense(layers[5], activation="tanh")(x)

    # Modelo final
    model = Model(inputs=inputs, outputs=outputs)

    # Compilar o modelo com otimizador Adam e perda configurável
    model.compile(
        loss=configs['model']['loss_function'],
        optimizer=Adam(learning_rate=learning_rate),
    )

    return model

# Função para carregar um modelo salvo
def load_network(filename):
    if os.path.isfile(filename):
        return load_model(filename)
    else:
        print(f'ERROR: "{filename}" file does not exist as an h5 model')
        return None

# Configurar callbacks para o treinamento
def get_callbacks(output_path):
    return [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(output_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

# Função para realizar a busca de hiperparâmetros
def tune_model():
    tuner = kt.Hyperband(
        build_network,
        objective='val_loss',
        max_epochs=50,
        factor=3,
        directory='hyperband_tuning',
        project_name='lstm_attention_model'
    )

    # Configurar dados fictícios para teste
    x_train = np.random.rand(1000, 50, 5)  # Dimensões iniciais padrão
    y_train = np.random.rand(1000, 1)
    x_val = np.random.rand(200, 50, 5)
    y_val = np.random.rand(200, 1)

    tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    # Melhor modelo encontrado
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Melhores hiperparâmetros:", best_hps.values)

    return tuner

# Exemplo de uso
if __name__ == "__main__":
    tuner = tune_model()

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    # Configurar dados fictícios para treinamento final
    x_train = np.random.rand(1000, best_hps['time_steps'], best_hps['features'])
    y_train = np.random.rand(1000, best_hps['output_dim'])
    x_val = np.random.rand(200, best_hps['time_steps'], best_hps['features'])
    y_val = np.random.rand(200, best_hps['output_dim'])

    # Caminho para salvar o melhor modelo
    output_path = "best_model_tuned.h5"

    # Treinamento com os melhores hiperparâmetros
    best_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=get_callbacks(output_path),
        verbose=1
    )
