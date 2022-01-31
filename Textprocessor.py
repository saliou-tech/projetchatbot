import re 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
import numpy as np

class TextProcessor():
    
    def __init__(self):
        pass
    
    def load_corpus(self,line_path,conversation_path):
       
        lines = open(line_path,encoding='utf-8',errors='ignore',).read().split('\n')
        conversations = open(conversation_path,encoding='utf-8',errors='ignore',).read().split('\n')

        idToLine={}
        for line in lines:
            _lines= line.split(' +++$+++ ')
            if len(_lines) ==5:
                idToLine[_lines[0]]=_lines[4]
        #print(idToLine['L665162 '])
        conversations_ids=[]
        for conversation in conversations[:-1]:
            _conversations= conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ","")
            conversations_ids.append(_conversations.split(','))

        questions=[]
        answers=[]

        for conversation in conversations_ids:
            #print(conversation[1])
            for i in range(len(conversation)-1):
                #print(conversation[i])
                questions.append(idToLine[conversation[i]])
                answers.append(idToLine[conversation[i+1]])

        # print(questions[1])
        # print(answers[1])
        sorted_ques=[]
        sorted_ans=[]
        for i in range(len(questions)):
            if(len(questions[i]))<13:
                sorted_ques.append(questions[i])
                sorted_ans.append(answers[i])
        return sorted_ques,sorted_ans
    
    def CleanText(self,text):
        text=text.lower()
        text=re.sub(r"i'am","i am",text)
        text=re.sub(r"he'e","he is",text)
        text=re.sub(r"she's","she is ",text)
        text=re.sub(r"it's","it is ",text)
        text=re.sub(r"what's","what is ",text)
        text=re.sub(r"that's","that is",text)
        text=re.sub(r"where's","where is ",text)
        text=re.sub(r"\'ve","have",text)
        text=re.sub(r"\'re","are",text)
        text=re.sub(r"\'d","would",text)
        text=re.sub(r"\'ll","will",text)
        text=re.sub(r"won't","will not",text)
        text=re.sub(r"can''t","can not",text)
        text=re.sub(r"[-()/#@;:,$^*!><!+=?^]","",text)
        text=re.sub(r"i'am","i am",text)
        return text
    def CleanQuestionAndAnswer(self,questions,responses):
        clean_ques=[]
        clean_ans=[]
        for line in questions:
            clean_ques.append(self.CleanText(line))
        for line in responses:
            clean_ans.append(self.CleanText(line))
            
        return clean_ques,clean_ans
    def wordToCount(self,questions,responses):
        word2Count={}
        #print(wordToCount)
        ################for response################
        for response in responses:
            for word in response.split():
                if word not in word2Count:
                    word2Count[word]=1
                else:
                    word2Count[word]+=1
       
        for question in questions:
            for word in question.split():
                if word not in word2Count:
                    word2Count[word]=1
                else:
                    word2Count[word]+=1
        return word2Count
    def getVocab(self,word2Count,reponses):
        ###  remove less frequent ###
        thresh = 5
        vocab = {}
        word_num = 0
        for word, count in word2Count.items():
            if count >= thresh:
                vocab[word] = word_num
                word_num += 1
        for i in range(len(reponses)):
            reponses[i] = '<SOS> ' + reponses[i] + ' <EOS>'

        #print(clean_ans[1])

        tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
        x = len(vocab)
        for token in tokens:
            vocab[token] = x
            x += 1    
        return vocab
    def EncodeQuestionAndResponse(self,vocab,questions,responses):
        encoder_inp = []
        for line in questions:
            lst = []
            for word in line.split():
                if word not in vocab:
                    lst.append(vocab['<OUT>'])
                else:
                    lst.append(vocab[word])

            encoder_inp.append(lst)

        decoder_inp = []
        for line in responses:
            lst = []
            for word in line.split():
                if word not in vocab:
                    lst.append(vocab['<OUT>'])
                else:
                    lst.append(vocab[word])        
            decoder_inp.append(lst)
        print(decoder_inp[1])
       
        encoder_inp = pad_sequences(encoder_inp, 13, padding='post', truncating='post')
        decoder_inp = pad_sequences(decoder_inp, 13, padding='post', truncating='post')

        print(decoder_inp[1])
        print(decoder_inp[2])
        decoder_final_output = []
        for i in decoder_inp:
            decoder_final_output.append(i[1:]) 

        decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')
        decoder_final_output = to_categorical(decoder_final_output, len(vocab))



        print(decoder_final_output)
        
        return encoder_inp,decoder_inp,decoder_final_output

    def getQuestionGiveAnswer(self ,prepro1,vocab,enc_model,dec_model,dense,inv_vocab):

        prepro1 = self.CleanText(prepro1)
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

        print("chatbot attention : ", decoded_translation )

                    

