For Rasa.
1. rasa_data_process.py used for all dataset processing operations based on 6 different dimensions.

2. rasa_NLU.py used for rasa NLU agent processing, including agent training, test and evaluation.

3. rasa_run.py used for final one-click loop, so just run this file, we can get final evalution result based on all combinations of dimensions.

4. All result models are svaed in file models, and all result are svaed in file rasa_result.

5. Final evaluation result is in './Rasa/rasa_result/rasa_eval_result.csv'.


For Dialogflow.
1. dialogflow_NLU.py used for dialogflow NLU agent processing, including agent training, test and evaluation, and also all dataset processing operations based on 6 different dimensions.

2. dialogflow_run.py also used for final one-click loop, so just run this file, we can get final evalution result based on all combinations of dimensions.

3. All result are svaed in file dialogflow_result, since we run the file on both GCP and uni server, so we get 2 result respectively.

4. Final evaluation result is in './Dialogflow/dialogflow_result/dialogflow_result_final.csv'.

5. Since Googlo has free API service request limit for 180 times per minutes, so we have to train and evaluate NLU agent no more than 180 times per minute and rest for 30s. 