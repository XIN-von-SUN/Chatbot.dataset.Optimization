import rasa_data_process 
import rasa_NLU 
import pandas as pd
import os

def rasa_nlu_loop(data_file, write_file, test_data, dimensions):
    '''
    This function will generate the sub-dataset as training set based on the chosen dimensions, 
    and then train the rasa NLU models based on these sub-datasets,
    Finally evaluate the models getting the corresponding evaluation metric values.
    ''' 
    eval_metrics = []

    if os.path.exists('./rasa_result/model_set.txt'):
        model_set = open('./rasa_result/model_set.txt','w+').readlines()
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

            file_name = str(dim1)+'-'+str(dim2)+'-'+str(dim3)+'-'+str(dim4)+'-'+str(dim5)+'-'+str(dim6) 
            subset_file = write_file + file_name + '.md'
            rasa_data_process.rasa_data_loop(data_file, subset_file, dimensions[i])

            # rasa NLU models training part.
            model_directory, model_name = "./models/nlu/", file_name
            train_data_path, test_data_path = subset_file, test_data #"./data/data_test.md"
            
            train_loop = True

            rasa_nlu = rasa_NLU.rasa_NLU(train_loop, model_directory, model_name, train_data_path, test_data_path)
            
            if rasa_nlu.train_loop:
                print("Start training......")
                rasa_nlu.NLU_train()
            
            # rasa NLU models evaluation part.
            eval_result, overall_accuracy, overall_f1_score, overall_precision = rasa_nlu.NLU_evaluation()
            eval_metrics.append([file_name, dim1,dim2,dim3,dim4,dim5,dim6, eval_result, overall_accuracy, overall_f1_score, overall_precision])
    
            result = pd.DataFrame(eval_metrics, columns=['model_name', 'dim1_rm', 'dim2_sent_len', 'dim3_sent_num', 'dim4_pattern', 'dim5_SDPs', 'dim6_keywords', 'eval_result', 'overall_accuracy', 'overall_f1_score', 'overall_precision'])
            
            result.to_csv('./rasa_result/rasa_eval_result.csv')   # have to remove all previous result.csv at first

            model_set.append([dim1,dim2,dim3,dim4,dim5,dim6])
            f = open('./rasa_result/model_set.txt','w+')
            f.write(str(model_set))
            f.close()





if __name__=='__main__':

    data_file = './data/data_full.json'
    write_file = './rasa_result/sub_datasets/'    #Downloads/oos-eval-master/data/data_full.json'
    test_data = './data/df_test.md'

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

    rasa_nlu_loop(data_file, write_file, test_data, dimensions)

    