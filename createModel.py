
import re 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input

class CreateModel:

    def __init__(self):
        pass

    def create_model(self,vocab,enc_inp,dec_inp):

        #enc_inp = Input(shape=(13, ))
        #dec_inp = Input(shape=(13, ))

        VOCAB_SIZE = len(vocab)
        embed = Embedding(VOCAB_SIZE+1, output_dim=50, 
                        input_length=13,
                        trainable=True                  
                        )


        enc_embed = embed(enc_inp)
        enc_lstm = LSTM(400, return_sequences=True, return_state=True)
        enc_op, h, c = enc_lstm(enc_embed)
        enc_states = [h, c]

        dec_embed = embed(dec_inp)
        dec_lstm = LSTM(400, return_sequences=True, return_state=True)
        dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

        dense = Dense(VOCAB_SIZE, activation='softmax')

        dense_op = dense(dec_op)

        model = Model([enc_inp, dec_inp], dense_op)

        #model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')


        enc_model = Model([enc_inp], enc_states)



        # decoder Model
        decoder_state_input_h = Input(shape=(400,))
        decoder_state_input_c = Input(shape=(400,))

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


        decoder_outputs, state_h, state_c = dec_lstm(dec_embed , 
                                            initial_state=decoder_states_inputs)


        decoder_states = [state_h, state_c]



        dec_model = Model([dec_inp]+ decoder_states_inputs,
                                            [decoder_outputs]+ decoder_states)
        return model,enc_model,dec_model,dense

        #model.fit([encoder_inp, decoder_inp],decoder_final_output,epochs=15)
        