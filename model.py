import tensorflow as tf
from tensorflow.keras.models import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
# Import the CRF layer. Note: This requires the 'keras-contrib' package.
from keras_contrib.layers import CRF 

def build_bilstm_crf_model(max_len, n_words, n_tags, embedding_dim=100, lstm_units=128, dropout_rate=0.1):
    """
    Builds the BiLSTM-CRF model for sequence labeling.
    """
    
    # 1. Input Layer
    word_input = Input(shape=(max_len,), dtype='int32', name='word_input')
    
    # 2. Embedding Layer
    # mask_zero=True is crucial for handling padded sequences correctly
    model = Embedding(
        input_dim=n_words, 
        output_dim=embedding_dim, 
        input_length=max_len, 
        mask_zero=True
    )(word_input)
    
    model = Dropout(dropout_rate)(model)

    # 3. Bi-LSTM Layer
    # return_sequences=True is required because we need output for every input step
    model = Bidirectional(
        LSTM(units=lstm_units, 
             return_sequences=True, 
             recurrent_dropout=dropout_rate)
    )(model)
    
    # 4. TimeDistributed Dense Layer
    # Reduces the dimension before the CRF layer
    model = TimeDistributed(Dense(lstm_units, activation="relu"))(model)
    
    # 5. CRF Layer
    crf = CRF(n_tags, name='crf_layer') 
    output = crf(model)
    
    # Final Model
    bilstm_crf_model = Model(word_input, output)
    
    # Compile the model with the CRF-specific loss and metrics
    bilstm_crf_model.compile(
        optimizer='rmsprop', # A good default optimizer
        loss=crf.loss_function, 
        metrics=[crf.accuracy]
    )
    
    return bilstm_crf_model