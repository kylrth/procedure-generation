from sklearn.feature_extraction.text import TfidfVectorizer
from dataset import *
import numpy as np
import asyncio
from evaluation.heuristic import Heuristic
# from nltk.corpus import stopwords
import spacy
from string import punctuation
from spacy.lang.en import stop_words

# constants
dataset_lcstep = "lcstep"
dataset_recipenlg = "recipenlg"
dataset_champ = "champ"
data_dir = './dataset/'




class TfIdf(Heuristic):
    train_data : list[str]
    dataset_name : str
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        proc_list = self.get_train_data()
        self.train_data = self.convert_procedures_to_string(proc_list)
        # import pdb; pdb.set_trace()
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = stop_words.STOP_WORDS
        self.punctuations = list(punctuation)
        self.tf_idf_vect = self.fit_tf_idf(self.train_data)
        
    def get_train_data(self) -> list[Procedure]:
        if self.dataset_name == dataset_lcstep:
            ds = LCStep(data_dir)
        elif self.dataset_name == dataset_recipenlg:
            ds = RecipeNLG(data_dir, n=10000)
        elif self.dataset_name == dataset_champ:
            ds = CHAMP(data_dir)
        else:
            raise NotImplementedError(f"unrecognized dataset")

        return ds.procedures(Split.TRAIN)

    def tokenize(self, sentence):
        sentence = self.nlp(sentence)
        # lemmatizing
        sentence = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in sentence ]
        # removing stop words
        sentence = [ word for word in sentence if word not in self.stop_words and word not in self.punctuations ]        
        return sentence

    def convert_procedures_to_string(self, proc_list: list[Procedure]) -> list[str]:
        out_str_list = []
        for procedure in proc_list:
            # proc_str = procedure.output + " using " + procedure._input
            # proc_str += "\n\n" + "\n".join("- " + s for s in procedure.steps)
            proc_str = "\n".join("- " + s for s in procedure.steps)
            out_str_list.append(proc_str)
        
        return out_str_list

    def fit_tf_idf(self, content):
        content = [self.tokenize(x.lower()) for x in content]
        vectorizer = TfidfVectorizer(preprocessor=' '.join)
        # import pdb; pdb.set_trace()
        X = vectorizer.fit(content)
        return vectorizer

    def convert_to_tf_idf_features(self, proc_str, vectorizer):
        doc_term_matrix = vectorizer.transform([self.tokenize(proc_str.lower())]).todense()
        # print(doc_term_matrix.shape)
        doc_term_matrix = np.squeeze(np.asarray(doc_term_matrix)) #Shape should be n_features now
        top_20_indices = np.argsort(doc_term_matrix)[-20:]
        # import pdb; pdb.set_trace()
        return vectorizer.get_feature_names_out()[top_20_indices]

    def evaluate(self, gold: Procedure, generated: list[str]):
        #Get important words
        
        gen_imp_words = self.convert_to_tf_idf_features(format_steps(generated), self.tf_idf_vect).tolist()
        print("Got important words in generated sequence")
        gold_imp_words = self.convert_to_tf_idf_features(gold.format_steps(), self.tf_idf_vect).tolist()
        print("Got important words in gold sequence")
        # import pdb; pdb.set_trace()
        count = 0
        for word in gen_imp_words:
            if word in gold_imp_words:
                count += 1
        print("Returning important words overlap score")
        return (count/len(gen_imp_words))