import pprint
import rasa.nlu
import rasa.core
import spacy
spacy.load("en")
from rasa.nlu.training_data import load_data
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Trainer, Interpreter
from rasa.nlu import config
from rasa.nlu.test import run_evaluation


class rasa_NLU:

    def __init__(self, train_loop, model_directory, model_name, train_data_path, test_data_path):
        self.train_loop = train_loop
        self.model_directory = model_directory
        self.model_name = model_name
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path


    def NLU_train(self):
        # loading the nlu training samples
        training_data = load_data(self.train_data_path)    #("./data/data_train.md")
        # trainer to educate our pipeline
        trainer = Trainer(config.load("config.yml"))
        # train the model!
        interpreter = trainer.train(training_data)
        # store it for future use
        model = trainer.persist(self.model_directory, fixed_model_name=self.model_name)    #("./models/nlu", fixed_model_name="current")


    def NLU_test(self, query):
        # Here just load the trained model directly

        model_path = self.model_directory + self.model_name
        interpreter = Interpreter.load(model_path)    #("./models/nlu/current")
        result = interpreter.parse(query)
        pred_intent = result['intent']['name']
        print('The resolved intent is: ', pred_intent)

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(result)

        return result, pred_intent


    def NLU_evaluation(self):

        #evaluation = run_evaluation("./data/data_test.md", model_directory)
        model_path = self.model_directory + self.model_name
        eval_result = run_evaluation(self.test_data_path, model_path)
        overall_accuracy = eval_result['intent_evaluation']['accuracy']
        overall_f1_score = eval_result['intent_evaluation']['f1_score']
        overall_precision = eval_result['intent_evaluation']['precision']

        print('The accuracy is: {}. The f1_score is: {}. The precision is: {}.'.format(overall_accuracy,overall_f1_score,overall_precision))
        
        return eval_result, overall_accuracy, overall_f1_score, overall_precision


if __name__=='__main__':

    model_directory, model_name = "./models/nlu/", "current"
    train_data_path, test_data_path = "./data/data_train.md", "./data/data_test.md"
    test_query = 'Hello, how are you!'
    train_loop = False

    rasa_nlu = rasa_NLU(train_loop, model_directory, model_name, train_data_path, test_data_path)
    
    while rasa_NLU.train_loop:
        print("Start training......")
        rasa_nlu.NLU_train()
    
    rasa_nlu.NLU_test(test_query)

    rasa_nlu.NLU_evaluation()





