import logging
from string import punctuation

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import stop_words

from dataset import CHAMP, LCStep, LinearProcedure, RecipeNLG, Split, format_steps

from .heuristic import Heuristic


# constants
dataset_lcstep = "lcstep"
dataset_recipenlg = "recipenlg"
dataset_champ = "champ"
data_dir = "./dataset/"


class TfIdf(Heuristic):
    train_data: list[str]
    dataset_name: str

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        proc_list = self.get_train_data()
        self.train_data = self.convert_procedures_to_string(proc_list)
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = stop_words.STOP_WORDS
        self.punctuations = list(punctuation)
        self.tf_idf_vect = self.fit_tf_idf(self.train_data)

    def get_train_data(self) -> list[LinearProcedure]:
        if self.dataset_name == dataset_lcstep:
            ds = LCStep(data_dir)
        elif self.dataset_name == dataset_recipenlg:
            ds = RecipeNLG(data_dir, n=10000)
        elif self.dataset_name == dataset_champ:
            ds = CHAMP(data_dir)
        else:
            raise NotImplementedError("unrecognized dataset")

        return ds.procedures(Split.TRAIN)

    def tokenize(self, sentence):
        sentence = self.nlp(sentence)
        # lemmatizing
        sentence = [
            word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_
            for word in sentence
        ]
        # removing stop words
        sentence = [
            word
            for word in sentence
            if word not in self.stop_words and word not in self.punctuations
        ]
        return sentence

    def convert_procedures_to_string(self, proc_list: list[LinearProcedure]) -> list[str]:
        out_str_list = []
        for procedure in proc_list:
            proc_str = "\n".join("- " + s for s in procedure.steps)
            out_str_list.append(proc_str)

        return out_str_list

    def fit_tf_idf(self, content):
        content = [self.tokenize(x.lower()) for x in content]
        vectorizer = TfidfVectorizer(preprocessor=" ".join)
        vectorizer.fit(content)
        return vectorizer

    def convert_to_tf_idf_features(self, proc_str, vectorizer):
        doc_term_matrix = vectorizer.transform([self.tokenize(proc_str.lower())]).todense()
        doc_term_matrix = np.squeeze(np.asarray(doc_term_matrix))  # Shape should be n_features now
        top_20_indices = np.argsort(doc_term_matrix)[-20:]
        return vectorizer.get_feature_names_out()[top_20_indices]

    def evaluate(
        self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]
    ) -> float:
        # Get important words

        gen_imp_words = self.convert_to_tf_idf_features(
            format_steps(generated), self.tf_idf_vect
        ).tolist()
        logger.debug("Got important words in generated sequence")
        gold_imp_words = self.convert_to_tf_idf_features(
            gold.format_steps(), self.tf_idf_vect
        ).tolist()
        logger.debug("Got important words in gold sequence")
        count = 0
        for word in gen_imp_words:
            if word in gold_imp_words:
                count += 1
        logger.debug("Returning important words overlap score")
        return count / len(gen_imp_words)

    async def aevaluate(self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]):
        raise NotImplementedError
