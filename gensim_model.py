"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
from typing import Any
from gensim.models import Word2Vec
from dataloader import gensim_dataloader
import gensim  # for other ops
import argparse  # for parsing the options
import os  # for os ops


def train_model(args) -> None:
    """Function to train word2vec model

    Args:
        args (Any): args to create the word2vec model and process them
    """

    # load data
    sentences = gensim_dataloader(args.path)

    print("Starting Model training")
    # init the model
    model = Word2Vec(sentences, sg=1, min_count=args.min_count)

    # create a path to save the model
    _path = os.path.split(args.path_to_saved_model)
    if len(_path) > 1:
        os.makedirs(os.path.join(*_path[:-1]), exist_ok=True)

    # save the model
    model.save(args.path_to_saved_model)


def predict_word_similarity(args:Any)-> None:
    """Function to predict the word similarity for word2vec model

    Args:
        args (Any): argparser for loading trained model and topn words similarity
    """
    model = Word2Vec.load(args.path)
    print(model.wv.most_similar(args.word, topn=args.top))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Script to train and predict the word2vec model using gensim")

    subparser = argparser.add_subparsers(help="train,predict")

    train = subparser.add_parser("train",
                                 description="parser to train the model")
    predict = subparser.add_parser(
        "predict",
        description="parser to predict embedding from the trained model")

    train.add_argument("--path", help="path to data file", dest="path")
    train.add_argument("--path_to_saved_model",
                       help="path to save model file",
                       dest="path_to_saved_model")
    train.add_argument("--N",
                       help="min_count size for the trainer,default 5",
                       dest="min_count",
                       type=int,
                       default=5)

    train.set_defaults(func=train_model)

    predict.add_argument("--path",
                         help="path to saved model file",
                         dest="path")
    predict.add_argument("--word",
                         help="word who's similarity needs to be computed",
                         dest="word")
    predict.add_argument("--top",
                         help="topn word similarities to load, default 10",
                         type=int,
                         default=10,
                         dest="top")

    predict.set_defaults(func=predict_word_similarity)

    args = argparser.parse_args()

    args.func(args)
