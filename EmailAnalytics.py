from typing import Dict, List, Tuple

import re
import string
import os
import networkx as nx
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from textblob import TextBlob

# NLTK and necessary downloads from it
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")


class Email:
    """
    Practice 2 - Digital email analytics.

    Implement this class while respecting exactly:
    - the class name
    - the method names
    - the parameters of each method
    - the general output type requested in the docstrings

    You may add private helper functions if needed.
    """

    REQUIRED_COLUMNS = [
        "email_id",
        "date",
        "sender",
        "recipients",
        "cc",
        "subject",
        "body",
    ]

    def __init__(self, csv_path: str):
        """
        Class constructor.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file with the emails.

        Minimum expected tasks
        ------------------------
        1. 1. Store the path as an attribute.
        2. 2. Prepare the attributes that will be used during the assignment:
           - self.df
           - self.graph
           - self.dictionary
           - self.corpus
           - self.lda_model
        """
        self.csv_path = csv_path
        self.df = None
        self.graph = None
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

    def load_data(self) -> pd.DataFrame:
        """
        Read the CSV, validate columns and prepare the base DataFrame.

        Minimum requirements:
        - Load the CSV into a pandas DataFrame.
        - Verify that the required columns exist.
        - Convert 'date' to datetime.
        - Replace null values in 'cc', 'subject' and 'body' with an empty string.
        - Create a 'text' column combining subject and body.
        - Store the DataFrame in self.df.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])
        df["cc"] = df["cc"].fillna("")
        df["subject"] = df["subject"].fillna("")
        df["body"] = df["body"].fillna("")
        df["text"] = df["subject"] + " " + df["body"]

        self.df = df
        return self.df

    def build_interaction_graph(self, include_cc: bool = True) -> nx.DiGraph:
        """
        Build a directed graph of interactions between senders and recipients.

        Minimum requirements:
        - Crear un nx.DiGraph.
        - Add an edge from the sender to each main recipient.
        - If include_cc=True, also include addresses in cc.
        - If an edge already exists, increment its weight.
        - Store the graph in self.graph.

        Notas
        -----
        - In 'recipients' and 'cc' there may be multiple addresses separated by ';'.

        Returns
        -------
        nx.DiGraph
            Directed graph with weighted edges.
        """
        if self.df is None:
            raise RuntimeError("DataFrame is not loaded. Call load_data() first.")

        self.graph = nx.DiGraph()

        for _, row in self.df.iterrows():
            sender = row["sender"].strip().lower()

            # Parse recipients and cc
            recipients = [
                r.strip().lower()
                for r in str(row["recipients"]).replace(",", ";").split(";")
                if r.strip()
            ]

            cc_list = []

            if include_cc and row["cc"]:
                cc_list = [
                    r.strip().lower()
                    for r in str(row["cc"]).replace(",", ";").split(";")
                    if r.strip()
                ]

            for recipient in recipients + cc_list:
                if self.graph.has_edge(sender, recipient):
                    self.graph[sender][recipient]["weight"] += 1
                else:
                    self.graph.add_edge(sender, recipient, weight=1)

        return self.graph

    def analyze_sentiment(self, text_column: str = "text") -> pd.DataFrame:
        """
        Compute sentiment with TextBlob.

        Minimum requirements:
        - Apply TextBlob to the specified column.
        - Create the following columns:
          * polarity (float)
          * subjectivity (float)
          * sentiment_label (str)
        - The labeling criterion is free, but it must generate at least the classes
          'positive', 'neutral' and 'negative'.

        Returns
        -------
        pd.DataFrame
            DataFrame updated with the sentiment columns.
        """
        if self.df is None:
            raise RuntimeError("DataFrame is not loaded. Call load_data() first.")

        if text_column not in self.df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        def get_sentiment_label(polarity: float) -> str:
            if polarity <= -0.05:
                return "negative"
            elif polarity >= 0.05:
                return "positive"
            else:
                return "neutral"

        self.df["polarity"] = self.df[text_column].apply(
            lambda t: TextBlob(str(t)).sentiment.polarity
        )

        self.df["subjectivity"] = self.df[text_column].apply(
            lambda t: TextBlob(str(t)).sentiment.subjectivity
        )

        self.df["sentiment_label"] = self.df["polarity"].apply(get_sentiment_label)

        return self.df

    def preprocess_text_for_lda(self, text: str) -> List[str]:
        """
        Preprocess a text for LDA.

        Student freedom:
        - You may decide how to tokenize.
        - You may remove stopwords.
        - You may apply stemming or lemmatization if desired.

        Restriction:
        - It must return a list of tokens ready to build the corpus.

        Returns
        -------
        List[str]
        """
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"https\S+|www\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove punctuation and digits
        text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
        text = re.sub(r"\d+", "", text)

        # Tokenize
        tokens = text.split()

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]

        # Remove short tokens
        tokens = [t for t in tokens if len(t) >= 3]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def train_topic_model(
        self, num_topics: int = 3, passes: int = 15, random_state: int = 42
    ) -> Tuple[LdaModel, Dictionary, List[List[tuple]]]:
        """
        Train an LDA model with gensim.

        Minimum requirements:
        - Preprocess the texts.
        - Build a gensim Dictionary.
        - Build the bag-of-words corpus.
        - Train an LdaModel.
        - Store dictionary, corpus and lda_model as attributes.

        Returns
        -------
        tuple
            (lda_model, dictionary, corpus)
        """
        if self.df is None:
            raise RuntimeError("DataFrame is not loaded. Call load_data() first.")

        # Preprocess all texts
        tokenized = self.df["text"].apply(self.preprocess_text_for_lda).tolist()

        # Build gensim Dictionary and filter extremes
        self.dictionary = Dictionary(tokenized)
        self.dictionary.filter_extremes(no_below=1, no_above=0.9)

        # Build bag-of-words corpus
        self.corpus = [self.dictionary.doc2bow(tokens) for tokens in tokenized]

        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=random_state,
        )

        return self.lda_model, self.dictionary, self.corpus

    def assign_topics(self) -> pd.DataFrame:
        """
        Assign a dominant topic to each email.

        Minimum requirements:
        - Use self.lda_model and self.corpus.
        - Create at least the following columns:
          * dominant_topic (int)
          * topic_keywords (str)

        Returns
        -------
        pd.DataFrame
            DataFrame with topic assignment.
        """
        pass

    def get_topic_report(self, topn_words: int = 5) -> pd.DataFrame:
        """
        Generate a structured summary by topic.

        Minimum expected format of the output DataFrame:
        - topic_id
        - keywords
        - num_emails
        - mean_polarity

        You may add extra columns if they add value.

        Returns
        -------
        pd.DataFrame
        """
        pass

    def get_emails_by_sender(self, sender: str) -> pd.DataFrame:
        """
        Return the emails sent by a specific sender.
        """
        pass

    def get_emails_by_topic(self, topic_id: int) -> pd.DataFrame:
        """
        Return the emails associated with a specific topic.
        """
        pass

    def graph_metrics(self) -> Dict[str, float]:
        """
        Return basic graph metrics.

        Minimum expected format:
        {
            "num_nodes": ...,
            "num_edges": ...,
            "density": ...
        }
        """
        pass
