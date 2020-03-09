import dialogflow_v2 as dialogflow
import os
import numpy as np
import pandas as pd
import string, re
import spacy
import random
import networkx as nx
from rake_nltk import Rake
import time

credentials_file = 'devbot-qludto-c49ed3f01f08.json'  
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

#from google.oauth2 import service_account
#credentials = service_account.Credentials.from_service_account_file(credentials_file)


def create_intent(project_id, display_name, training_phrases_parts):
    """Create an intent of the given intent type."""

    intents_client = dialogflow.IntentsClient()
    parent = intents_client.project_agent_path(project_id)

    for i in range(len(display_name)):
        training_phrases = []
        for training_phrases_part in training_phrases_parts[i]:
            part = dialogflow.types.Intent.TrainingPhrase.Part(
                text=training_phrases_part)
            # Here we create a new training phrase for each provided part.
            training_phrase = dialogflow.types.Intent.TrainingPhrase(parts=[part])
            training_phrases.append(training_phrase)

        #text = dialogflow.types.Intent.Message.Text(text=message_texts)
        #message = dialogflow.types.Intent.Message(text=text)
        try:
            intent = dialogflow.types.Intent(
                display_name=display_name[i],
                training_phrases=training_phrases)

            response = intents_client.create_intent(parent, intent)
            #print('Intent created: {}'.format(response))

        except: 
            print('meet problem in intent: ', display_name[i])
            pass

        continue



def train_agent(project_id):
    client = dialogflow.AgentsClient()

    parent = client.project_path(project_id)

    response = client.train_agent(parent)



def get_intent_id(project_id, name):
    #name = display_name[2]

    intents_client = dialogflow.IntentsClient()
    parent = intents_client.project_agent_path(project_id)

    intents = intents_client.list_intents(parent)

    intent_names = [
        intent.name for intent in intents
        if intent.display_name == name]

    intent_ids = [
        intent_name.split('/')[-1] for intent_name
        in intent_names]

    return intent_ids



def detect_intent(project_id, text_to_be_analyzed):
    DIALOGFLOW_PROJECT_ID = project_id
    DIALOGFLOW_LANGUAGE_CODE = 'en'
    SESSION_ID = '1'  #SESSION_ID

    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)

    #text_to_be_analyzed = "transfer money"
    text_to_be_analyzed = text_to_be_analyzed #"set alarm"

    text_input = dialogflow.types.TextInput(text=text_to_be_analyzed, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)

    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
        return response.query_result.intent.display_name

    except:  #InvalidArgument:
        pass

    '''
    print("Query text:", response.query_result.query_text)
    print("Detected intent:", response.query_result.intent.display_name)
    print("Detected intent confidence:", response.query_result.intent_detection_confidence)
    #print("Fulfillment text:", response.query_result.fulfillment_text)
    '''



def eval_intent(project_id, test_data, write_file):  # input test data shoulb be in format .csv
    '''DIALOGFLOW_PROJECT_ID = project_id
    DIALOGFLOW_LANGUAGE_CODE = 'en'
    SESSION_ID = '1'

    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
    '''

    for i in range(int(len(test_data)/2)):
        if i != 1 and i % 150 == 1:
            time.sleep(65)
            print('Evaluating intent need to sleep for 65s!')

        try:
            pred_intent = detect_intent(project_id, test_data.loc[i, 'Phrase'])
            #print('predicted intent is: ', pred_intent)
            test_data.loc[i, 'Pred_intent'] = pred_intent
        except: pass

        continue

    for i in range(int(len(test_data)/2), len(test_data)):
        if i % 150 == 1:
            time.sleep(65)
            print('Evaluating intent need to sleep for 65s!')

        try:
            pred_intent = detect_intent(project_id, test_data.loc[i, 'Phrase'])
            #print('predicted intent is: ', pred_intent)
            test_data.loc[i, 'Pred_intent'] = pred_intent
        except: pass

        continue
   
    test_data.dropna(axis=0, how='any', inplace=True)  
    test_data = test_data.reset_index(drop=True)
    test_data.to_csv('./dialogflow_result/test_data_result.csv')
    print('test_data length is: ', len(test_data))
    print('Evaluation is done!')
    
    intent_set = list(set(list(test_data.Intent.unique()) + list(test_data.Pred_intent.unique())))
    con_mat =  np.zeros([len(intent_set), len(intent_set)])  # True * Prediction
    print('con_mat shape is: ', len(test_data))

    for i in range(len(test_data)):
        con_mat[intent_set.index(test_data.loc[i, 'Intent']), intent_set.index(test_data.loc[i, 'Pred_intent'])] += 1
        np.save(write_file+'/con_mat.npy', con_mat)
    print('confusion matrix is done!')

    
    TP,TN,FP,FN,accuracy,precision,recall,f1 = 0,0,0,0,0,0,0,0
    precision_sum, recall_sum, f1_sum = 0,0,0
    d = 0
    for i in range(len(con_mat)-1):
        TP = con_mat[i][i]
        FN = (np.sum(con_mat[i, :i]) + np.sum(con_mat[i, i+1:]))
        FP = (np.sum(con_mat[:i, i]) + np.sum(con_mat[i+1:, i]))
        TN = (np.sum(con_mat) - TP - FN - FP)

        if  (TP + FP) != 0 and  (TP + FN) != 0:
            d += 1
            accuracy += TP / (TP + TN + FP + FN)

            precision = TP / (TP + FP)
            precision_sum += precision

            recall = TP / (TP + FN)
            recall_sum += recall

            f1 = (2 * precision * recall / (precision + recall))
            f1_sum += f1
    #print(accuracy,precision,recall,f1)

    TP = con_mat[len(con_mat)-1, len(con_mat)-1]
    FN = np.sum(con_mat[len(con_mat)-1, :len(con_mat)-1])
    FP = np.sum(con_mat[:len(con_mat)-1, len(con_mat)-1])
    TN = (np.sum(con_mat) - TP - FN - FP)

    if  (TP + FP) != 0 and  (TP + FN) != 0:
        d += 1
        accuracy += TP / (TP + TN + FP + FN)

        precision = TP / (TP + FP)
        precision_sum += precision

        recall = TP / (TP + FN)
        recall_sum += recall

        f1 = (2 * precision * recall / (precision + recall))
        f1_sum += f1

    print(accuracy, precision_sum/d, recall_sum/d, f1_sum/d)

    overall_accuracy, overall_precision, overall_f1 = accuracy, precision_sum/d, f1_sum/d
    
    return overall_accuracy, overall_precision, overall_f1




def delete_all_intents(project_id):
    intents_client = dialogflow.IntentsClient()
    parent = intents_client.project_agent_path(project_id)

    intents = intents_client.list_intents(parent)
    intent_set_bf = [i.display_name for i in intents]
    print('Intent before delete: ', len(intent_set_bf))

    if len(intent_set_bf) != 0:
        intents = intents_client.list_intents(parent)
        response = intents_client.batch_delete_intents(parent, intents)
    
    print('-'*20)
    intents = intents_client.list_intents(parent)
    intent_set_aft = [i.display_name for i in intents]
    print('Intent after delete: ', len(intent_set_aft))




def dialogflow_sub_dataset(data_file, dim1_stop, dim2_length, dim3_reduce, dim4_grammar, dim5_sdp, dim6_keywords):
    #data_file = '/users/xinsun/Downloads/oos-eval-master/data/data_full.json'
    df = pd.read_json(data_file, typ='series')
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

    return df_final



def dialogflow_data_loop(data_file, write_file, dimensions):

    sub_data = dialogflow_sub_dataset(data_file, dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5])
    intent_list = list(sub_data.Intent.unique())
    phrase_list = [sub_data[sub_data.Intent==intent]['Phrase'].values for intent in intent_list]
        
    file_name = str(dimensions[0])+'-'+str(dimensions[1])+'-'+str(dimensions[2])+'-'+str(dimensions[3])+'-'+str(dimensions[4])+'-'+str(dimensions[5])
    subset_file = write_file + file_name + '.txt'
    print(subset_file)
    
    with open(subset_file, 'w+') as f:   
        f.write(str(intent_list))  
        f.write(str(phrase_list))  
        f.close()

    return intent_list, phrase_list, file_name



def dialogflow_nlu_loop(project_id, data_file, write_file, test_data, dimensions):
    '''
    This function will generate the sub-dataset as training set based on the chosen dimensions, 
    and then train the dialogflow NLU models based on these sub-datasets,
    Finally evaluate the models getting the corresponding evaluation metric values.
    ''' 

    # load the test data
    #ids = np.random.randint(0,5400,3000)
    #df_test = pd.read_csv(test_data, index_col=0).loc[ids].reset_index(drop=True)
    df_test = pd.read_csv(test_data, index_col=0).loc[::2].reset_index(drop=True)

    eval_metrics = []

    if os.path.exists('./dialogflow_result/model_set.txt'):
        model_set = eval(open('./dialogflow_result/model_set.txt', 'r').readlines()[0])
    else:
        model_set = []

    # sub-dataset part.
    for i in range(len(dimensions)):  # range(len(dimensions))

        dim1 = dimensions[i][0]
        dim2 = dimensions[i][1] #if dimensions[i][1] else bool(dimensions[i][1])
        dim3 = dimensions[i][2] #if dimensions[i][2] else bool(dimensions[i][2])
        dim4 = dimensions[i][3] #if dimensions[i][3] else bool(dimensions[i][3])
        dim5 = dimensions[i][4]
        dim6 = dimensions[i][5]

        # In order to avoiding the same training model config.
        if [dim1,dim2,dim3,dim4,dim5,dim6] not in model_set:

            intent_list, phrase_list, sub_file_name = dialogflow_data_loop(data_file, write_file, dimensions[i])
            
            print('_'*25, 'One dimension sub dataset is completed!')
            print('number of intents is:', len(intent_list))
            print('number of intents phrase is:', len(phrase_list))
            print(sub_file_name)


            # dialogflow NLU models training part.
            # delete all the intents of current agent model
            client = dialogflow.IntentsClient()
            parent = client.project_agent_path(project_id)
            intents = client.list_intents(parent)
            response = client.batch_delete_intents(parent, intents)
            print('The agent is reset!')


            # firstly create all intents
            intents_client = dialogflow.IntentsClient()
            parent = intents_client.project_agent_path(project_id)

            display_name, training_phrases_parts = intent_list, phrase_list
            
            for i in range(len(display_name)):

                if i != 1 and i % 45 == 1:
                    time.sleep(60)
                    print('Creating intent need to sleep for 60s!')

                try:
                    training_phrases = []
                    for training_phrases_part in training_phrases_parts[i]:
                        part = dialogflow.types.Intent.TrainingPhrase.Part(
                            text=training_phrases_part)
                        # Here we create a new training phrase for each provided part.
                        training_phrase = dialogflow.types.Intent.TrainingPhrase(parts=[part])
                        training_phrases.append(training_phrase)

                    text = dialogflow.types.Intent.Message.Text(text=message_texts)
                    message = dialogflow.types.Intent.Message(text=text)

                    intent = dialogflow.types.Intent(
                        display_name=display_name[i],
                        training_phrases=training_phrases,
                        messages=[message])

                    response = intents_client.create_intent(parent, intent)
                except: pass
                
                continue
            
            time.sleep(30)
            print('The intents are all created successfully!')
            #print('Intent created: {}'.format(response))

            
            # then training the agent model
            client = dialogflow.AgentsClient()
            parent = client.project_path(project_id)
            response = client.train_agent(parent) 
            time.sleep(30)
            print('The agent is trained successfully!')    

        
            # dialogflow NLU models evaluation part.
            overall_accuracy, overall_precision, overall_f1_score = eval_intent(project_id, df_test, write_file)
            eval_metrics.append([sub_file_name, dim1,dim2,dim3,dim4,dim5,dim6, overall_accuracy, overall_precision, overall_f1_score])

            result = pd.DataFrame(eval_metrics, columns=['model_name', 'dim1_rm', 'dim2_sent_len', 'dim3_sent_num', 'dim4_pattern', 'dim5_SDPs', 'dim6_keywords', 'overall_accuracy', 'overall_precision', 'overall_f1_score'])
            result.to_csv('./dialogflow_result/dialogflow_eval_result.csv')   # have to remove all previous result.csv at first
            time.sleep(30)

            model_set.append([dim1,dim2,dim3,dim4,dim5,dim6])
            f = open('./dialogflow_result/model_set.txt','w+')
            f.write(str(model_set))
            f.close()
            
            print('The agent is evaluated successfully!')


            '''
            # delete all the intents of current agent model
            client = dialogflow.IntentsClient()
            parent = client.project_agent_path(project_id)
            intents = client.list_intents(parent)
            response = client.batch_delete_intents(parent, intents)
            print('The agent is reset!')
            '''




if __name__=='__main__':

    data_file = './data/data_full.json'
    write_file = './dialogflow_result/sub_datasets/'    #Downloads/oos-eval-master/data/data_full.json'
    test_data = './data/df_test.csv'
    project_id = 'devbot-qludto'
    language_code = ''
    message_texts = None #'delete the value'

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

    dialogflow_nlu_loop(project_id, data_file, write_file, test_data, dimensions[:1])
    

