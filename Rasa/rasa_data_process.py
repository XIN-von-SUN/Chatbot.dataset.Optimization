import pandas as pd
import numpy as np
import string, re
import spacy
import random
import networkx as nx
from rake_nltk import Rake


class rasa_data:

    def __init__(self, data_file):
        self.data_file = data_file


    def rasa_data_process(self, df, write_file):
        '''
        Transfer the data to the rasa training data format.
        '''

        '''
        # read_file = '/users/xinsun/Downloads/oos-eval-master/data/data_full.json'
        df = pd.read_json(self.data_file, typ='series')

        df_oos_train = pd.DataFrame(df['oos_train'], columns=['Phrase', 'Intent'])
        df_oos_test = pd.DataFrame(df['oos_test'], columns=['Phrase', 'Intent'])
        df_oos_val = pd.DataFrame(df['oos_val'], columns=['Phrase', 'Intent'])
        df_train = pd.DataFrame(df['train'], columns=['Phrase', 'Intent'])
        df_test = pd.DataFrame(df['test'], columns=['Phrase', 'Intent'])
        df_val = pd.DataFrame(df['val'], columns=['Phrase', 'Intent'])

        df_train['Phrase'] = '- ' + df_train.Phrase + '.\n'
        df_test['Phrase'] = '- ' + df_test.Phrase + '.\n'

        intents_train = list(df_train.Intent.unique())
        intents_test = list(df_test.Intent.unique())

        data_train = []
        for intent in intents_train:
            data_train.append('## intent:' + str(intent) + '\n')
            phrase = df_train[df_train.Intent==intent]['Phrase'].values
            phrase_txt = str()
            for i in phrase: phrase_txt += str(i)
            data_train.append(phrase_txt + '\n')
            
        data_test = []
        for intent in intents_test:
            data_test.append('## intent:' + str(intent) + '\n')
            phrase = df_test[df_test.Intent==intent]['Phrase'].values
            phrase_txt = str()
            for i in phrase: phrase_txt += str(i)
            data_test.append(phrase_txt + '\n')

        # with open('data_test.txt', 'w') as f:   
        with open(write_file, 'w') as f:   
            for i in data_test:
                f.write(str(i))  
        '''

        df['Phrase'] = '- ' + df.Phrase + '\n'
        intents = list(df.Intent.unique())

        data = []
        for intent in intents:
            data.append('## intent:' + str(intent) + '\n')
            phrase = df[df.Intent==intent]['Phrase'].values
            phrase_txt = str()
            for i in phrase: phrase_txt += str(i)
            data.append(phrase_txt + '\n')
        
        with open(write_file, 'w+') as f:   
            for i in data:
                f.write(str(i))  



    def rasa_sub_dataset(self, write_file, dim1_stop, dim2_length, dim3_reduce, dim4_grammar, dim5_sdp, dim6_keywords):
        #data_file = '/users/xinsun/Downloads/oos-eval-master/data/data_full.json'
        df = pd.read_json(self.data_file, typ='series')
        df_train = pd.DataFrame(df['train'], columns=['Phrase', 'Intent'])
        df_oov_train = pd.DataFrame(df['oos_train'], columns=['Phrase', 'Intent'])

        df_train = pd.concat([df_train, df_oov_train]).reset_index(drop=True)

        '''whether delete punctuation or stop words.'''
        if bool(dim1_stop): 
            '''
            text = df_train.Phrase.values
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            token_no_punctuation = [regex.sub('', i) for i in text]
            df_train['Phrase'] = token_no_punctuation
            '''
            
            STOP_WORDS = ['the', 'please', 'would', 'much', 'to', 'that', 'me', "'d", "'ll", "'m", "'re", "'s", "'ve", 'a', 'an']

            text = df_train.Phrase.values
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            text_no_punctuation = [regex.sub('', i) for i in text]

            nlp = spacy.load('en')
            phrase_tokens = nlp.pipe(text_no_punctuation, batch_size=10000)
            token_no_punctuation = []
            for phrase in phrase_tokens: token_no_punctuation.append(phrase)

            token_no_punctuation_stopwords = [[token for token in doc if str(token) not in STOP_WORDS] for doc in token_no_punctuation]

            text_no_punctuation_stopwords = []
            for text in token_no_punctuation_stopwords:
                phrase = str()
                for i in text:
                    phrase += (str(i) + ' ')
                text_no_punctuation_stopwords.append(phrase.strip())

            regex = re.compile('[%s]' % re.escape(string.punctuation))
            text_no_punctuation_stopwords = [regex.sub('', i) for i in text_no_punctuation_stopwords]

            df_train['Phrase'] = text_no_punctuation_stopwords


        '''control the length of training sentences.'''
        if bool(dim2_length):
            nlp = spacy.load('en')
            phrases = df_train['Phrase']
            tokens = nlp.pipe(phrases, batch_size=10000)

            phrase_tokens = []
            for phrase in tokens: phrase_tokens.append(phrase)

            df_train_length = []
            for i in phrase_tokens: df_train_length.append(str(i[:dim2_length])) 

            df_train['Phrase'] = df_train_length


        '''control the number of training sentences for each intents.'''
        if bool(dim3_reduce):
            idx_range= [(i*100, i*100+99) for i in range(150)]
            idx = []
            for i in range(150):
                idx.append([random.randint(idx_range[i][0], idx_range[i][1]) for _ in range(dim3_reduce)])

            index = []
            for i in idx: 
                for j in range(len(i)): index.append(i[j])

            df_train['Phrase_copy'] = df_train['Phrase']
            df_train['Phrase'] = None
            df_train.loc[index, 'Phrase'] = df_train.loc[index, 'Phrase_copy']
            df_train = df_train.drop(columns=['Phrase_copy'])


        '''choose different patterns of training sentences.'''
        if bool(dim4_grammar):
            df_train = df_train[df_train.Phrase.isnull()==False].reset_index(drop=True)
            nlp = spacy.load('en')
            phrases = df_train['Phrase']
            tokens = nlp.pipe(phrases, batch_size=10000)

            phrase_tokens = []
            for phrase in tokens: phrase_tokens.append(phrase)

            idx_question = []
            for i in range(len(df_train)):
                if str(phrase_tokens[i][0]) in ['what', 'why', 'who', 'how', 'can']:
                    idx_question.append(i)
            idx_statement = set(range(len(df_train))) - set(idx_question)

            df_train['Phrase_question'] = df_train.loc[idx_question, 'Phrase']
            df_train['Phrase_statement'] = df_train.loc[idx_statement, 'Phrase']

            if dim4_grammar == 'question': df_train['Phrase'] = df_train['Phrase_question']
            else: df_train['Phrase'] = df_train['Phrase_statement']


        
        '''using shortest dependency path of training phrases.'''
        if bool(dim5_sdp):
            df_train = df_train[df_train.Phrase.isnull()==False].reset_index(drop=True)
            
            text = df_train.Phrase.values
            nlp = spacy.load('en')
            phrase_tokens = nlp.pipe(text, batch_size=10000)

            phrase_SDPs = []
            error_phrase = []
            for idx, phrase in enumerate(phrase_tokens):
                try:
                    edges = []
                    for token in phrase:
                        for child in token.children:
                            edges.append(('{0}'.format(token.lower_),
                                        '{0}'.format(child.lower_))) 

                    words_set= []
                    for i in edges:
                        words_set.append(i[0])
                        words_set.append(i[1])
                    words_set = set(words_set)

                    for start in range(0,len(phrase),1):
                        if str(phrase[start]) in words_set:
                            #print(start)
                            entity1 = str(phrase[start]).lower()
                            break

                    for end in range(len(phrase)-1,-1,-1):
                        if str(phrase[end]) in words_set:
                            #print(end)
                            entity2 = str(phrase[end]).lower()
                            break

                    graph = nx.Graph(edges)
                    # Get the SDP.
                    token_SDP = nx.shortest_path(graph, source=entity1, target=entity2)

                    SDP = ''
                    for i in token_SDP: SDP+= (i+' ')
                    phrase_SDPs.append(SDP.strip())
                except:
                    error_phrase.append(idx)
                    pass 
                
                continue

            df_train = df_train.drop(index=error_phrase).reset_index(drop=True)

            df_train['Phrase'] = phrase_SDPs

        
        '''just using keywords of each training phrases.'''
        if bool(dim6_keywords):

            df_train = df_train[df_train.Phrase.isnull()==False].reset_index(drop=True)
            
            phrases = df_train.Phrase.values

            r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.

            token_keywords = []
            for phrase in phrases:
                r.extract_keywords_from_text(phrase)
                token_keywords.append(r.get_ranked_phrases())

            phrase_keywords = []
            for tokens in token_keywords:
                sent = ''
                for token in tokens:
                    sent += (token + ' ')
                phrase_keywords.append(sent.strip())

            df_train['Phrase'] = phrase_keywords
                    

        df_final = df_train[df_train.Phrase.isnull()==False].reset_index(drop=True)
        #print(df_final)
        print('len of final df:', len(df_final))

        self.rasa_data_process(df_final[['Phrase', 'Intent']], write_file) 



def rasa_data_loop(data_file, write_file, dimensions):

    rasa_dataset = rasa_data(data_file)
    rasa_dataset.rasa_sub_dataset(write_file, dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5])



if __name__=='__main__':

    data_file = './data/data_full.json'
    write_file = './rasa_result/sub_datasets_process_loop/'    #Downloads/oos-eval-master/data/data_full.json'
    
    # get the test dataset
    df = pd.read_json(data_file, typ='series')
    df_test = pd.DataFrame(df['test'], columns=['Phrase', 'Intent'])
    df_oov_test = pd.DataFrame(df['oos_test'], columns=['Phrase', 'Intent'])
    df_test = pd.concat([df_test, df_oov_test]).reset_index(drop=True)
    
    write_test_file = './data/df_test.md'
    rasa_test_data = rasa_data(data_file)
    rasa_test_data.rasa_data_process(df_test[['Phrase', 'Intent']], write_test_file) 


    '''
    dimensions = [[True, False, False, False, False, False], [True, False, 20, 'statement', True, True], [True, False, 50, 'question', False, False], 
            [True, 5, 50, 'statement', True, True], [True, 5, 100, False, False, True]] 
    '''
    dim1_punct = [True, False]
    dim2_length = [False, 5, 15]
    dim3_reduce = [False, 15, 50]
    dim4_grammar = [False, 'statement', 'question']
    dim5_sdp = [True, False]
    dim6_keywords = [True, False]

    dimensions = [] 
    for d1 in dim1_punct:
        for d2 in dim2_length:
            for d3 in dim3_reduce:
                for d4 in dim4_grammar:
                    for d5 in dim5_sdp:
                        for d6 in dim6_keywords:
                            if (d2 == False and d5 == True and d6 == False) or (d2 == False and d5 == False and d6 == True) or (d5 == False and d6 == False):
                                dimensions.append([d1, d2, d3, d4, d5, d6])
    
    
    for i in range(len(dimensions)):
        file_name = str(dimensions[i][0])+'-'+str(dimensions[i][1])+'-'+str(dimensions[i][2])+'-'+str(dimensions[i][3])+'-'+str(dimensions[i][4])+'-'+str(dimensions[i][5])
        
        subset_file = write_file + file_name + '.md'
        print(subset_file)

        rasa_data_loop(data_file, subset_file, dimensions[i])