from Textprocessor import TextProcessor
from createModel import CreateModel
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
import numpy as np
import re 
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint
 

class Chatbot:
    
    def __init__(self):
        self.corpus=TextProcessor()
        self.createmodel=CreateModel()
        self.questions,self.responses=self.corpus.load_corpus('movie_lines.txt','movie_conversations.txt')
        self.questions,self.responses=self.corpus.CleanQuestionAndAnswer(self.questions,self.responses)
        self.word_to_count=self.corpus.wordToCount(self.questions,self.responses)
        self.vocab=self.corpus.getVocab(self.word_to_count,self.responses)
        self.inv_vocab = {w:v for v, w in self.vocab.items()}
        self.encoder_inp,self.decoder_inp,self.decoder_final_output=self.corpus.EncodeQuestionAndResponse(self.vocab,self.questions,self.responses)
        #declarations des inputs 
        self.enc_inp = Input(shape=(13, ))
        self.dec_inp = Input(shape=(13, ))
        self.model,self.enc_model,self.dec_model,self.dense=self.createmodel.create_model(self.vocab,self.enc_inp,self.dec_inp )
        self.model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')



        # model_file="chatbot-model_test.hdf5"

        # checkpoint = ModelCheckpoint(model_file, 
        #                             monitor='val_loss', 
        #                             verbose=1, 
        #                             save_best_only=True, 
        #                             )

        # es = EarlyStopping(monitor='val_loss', 
        #                 mode='min',
        #                 patience=100)

        # callbacks_list = [checkpoint]


        # self.model.fit([self.encoder_inp, self.decoder_inp],self.decoder_final_output, 
        #                     epochs=20,
        #                     verbose=1,
        #                     shuffle=True,
        #                     callbacks=callbacks_list)

        self.model = load_model('chatbot.h5')




    # prepo1 is the question of the user 
    def getQuestionGiveAnswer(self ,prepro1,vocab,enc_model,dec_model,dense,inv_vocab):

        prepro1 = self.corpus.CleanText(prepro1)
        ## prepro1 = "hello"

        prepro = [prepro1]
        ## prepro1 = ["hello"]

        txt = []
        for x in prepro:
            # x = "hello"
            lst = []
            for y in x.split():

                ## y = "hello"
                try:
                    lst.append(vocab[y])
                    ## vocab['hello'] = 454
                except:
                    lst.append(vocab['<OUT>'])
            txt.append(lst)

        ## txt = [[454]]
        txt = pad_sequences(txt, 13, padding='post')

        ## txt = [[454,0,0,0,.........13]]

        stat = enc_model.predict( txt )

        empty_target_seq = np.zeros( ( 1 , 1) )
        ##   empty_target_seq = [0]


        empty_target_seq[0, 0] = vocab['<SOS>']
        ##    empty_target_seq = [255]

        stop_condition = False
        decoded_translation = ''

        while not stop_condition :

            dec_outputs , h, c= dec_model.predict([ empty_target_seq] + stat )
            decoder_concat_input = dense(dec_outputs)
            ## decoder_concat_input = [0.1, 0.2, .4, .0, ...............]

            sampled_word_index = np.argmax( decoder_concat_input[0, -1, :] )
            ## sampled_word_index = [2]

            sampled_word = inv_vocab[sampled_word_index] + ' '

            ## inv_vocab[2] = 'hi'
            ## sampled_word = 'hi '

            if sampled_word != '<EOS> ':
                decoded_translation += sampled_word  

            if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 13:
                stop_condition = True 

            empty_target_seq = np.zeros( ( 1 , 1 ) )  
            empty_target_seq[ 0 , 0 ] = sampled_word_index
            ## <SOS> - > hi
            ## hi --> <EOS>
            stat = [h, c]  

        print("chatbot response : ", decoded_translation )
        print("------------------------discussion--------------------")

        return decoded_translation

                    


# chatbot=Chatbot()
# prepro1 = ""
# while prepro1 != 'q':
#     prepro1  = input("you : ")
#     chatbot.getQuestionGiveAnswer(prepro1,chatbot.vocab,chatbot.enc_model,chatbot.dec_model,chatbot.dense,chatbot.inv_vocab)
    

