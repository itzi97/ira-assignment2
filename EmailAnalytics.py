from typing import Dict, List, Tuple

import re
import string
import os
import networkx as nx
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from textblob import TextBlob


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
        self.graph = None  # build_interaction_graph() -> directed graph
        self.dictionary = None  # train_topic_model() -> gensim token dictionary
        self.corpus = None  # train_topic_model() -> bag-of-words
        self.lda_model = None  # train_topic_model() -> trained LDA model

        # Download NLTK resources if not present
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

    def _require_dataframe(self):
        """Raise an error if load_data() has not been called yet."""
        if self.df is None:
            raise RuntimeError("DataFrame is not loaded. Call load_data() first.")

    def _require_model(self):
        """Raise an error if train_topic_model() has not been called yet."""
        if self.lda_model is None or self.corpus is None:
            raise RuntimeError("LDA model not trained. Call train_topic_model()")

    def _require_topics(self):
        """Raise an error if assign_topics() has not been called yet."""
        if "dominant_topic" not in self.df.columns:
            raise RuntimeError("Topics not assigned. Call assign_topics() first.")

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
        # Check if file exists before trying to read it
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Validate that all 7 required columns are present
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert date column from plain string
        df["date"] = pd.to_datetime(df["date"])

        # Fill NaN values in text-based columns with empty strings
        df["cc"] = df["cc"].fillna("")
        df["subject"] = df["subject"].fillna("")
        df["body"] = df["body"].fillna("")

        # Combined text column used by sentiment analysis and LDA
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
        # Ensure load_data() has been called before this method
        self._require_dataframe()

        # Initialize empty directed graph (emails have direction)
        self.graph = nx.DiGraph()

        for _, row in self.df.iterrows():
            # Normalize sender address
            sender = row["sender"].strip().lower()

            # Parse recipients: Normalize, split, strip
            recipients = [
                r.strip().lower()
                for r in str(row["recipients"]).replace(",", ";").split(";")
                if r.strip()
            ]

            # Parse cc addresses the same way, but only if include_cc=True
            cc_list = []
            if include_cc and row["cc"]:
                cc_list = [
                    r.strip().lower()
                    for r in str(row["cc"]).replace(",", ";").split(";")
                    if r.strip()
                ]

            # Add edge from sender to each recipient (and CC if included)
            # If edge exists: increment weight instead of adding duplicate
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
        # Ensure load_data() has called before this method
        self._require_dataframe()

        # Validate that column exists in DataFrame
        if text_column not in self.df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        # Converts numeric polarity to human-readable label
        def get_sentiment_label(polarity: float) -> str:
            if polarity <= -0.05:
                return "negative"
            elif polarity >= 0.05:
                return "positive"
            else:
                return "neutral"

        # Apply TextBlob to each email's text and extract polarity score
        self.df["polarity"] = self.df[text_column].apply(
            lambda t: TextBlob(str(t)).sentiment.polarity
        )

        # Extract subjectivity score from TextBlob
        self.df["subjectivity"] = self.df[text_column].apply(
            lambda t: TextBlob(str(t)).sentiment.subjectivity
        )

        # Map each polarity float to its categorical label using get_sentiment_label
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
        # Ensure load_data() has been called before this method
        self._require_dataframe()

        # Preprocess all texts to list of clean tokens
        tokenized = self.df["text"].apply(self.preprocess_text_for_lda).tolist()

        # Build gensim Dictionary and filter extremes
        self.dictionary = Dictionary(tokenized)
        self.dictionary.filter_extremes(no_below=1, no_above=0.9)

        # Build bag-of-words corpus (list of (token_id, count) pairs)
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
        # Ensure train_topic_model() has been called before this method
        self._require_model()

        dominant_topics = []
        topic_keywords = []

        for bow in self.corpus:
            # Get topic distribution for this document
            topic_dist = self.lda_model.get_document_topics(
                bow, minimum_probability=0.0
            )

            # Pick the dominant topic (highest probability)
            dominant = max(topic_dist, key=lambda x: x[1])
            dominant_topic_id = dominant[0]

            # Get top keywords for that topic
            keywords = self.lda_model.show_topic(dominant_topic_id, topn=5)
            keywords_str = ", ".join([word for word, _ in keywords])

            dominant_topics.append(dominant_topic_id)
            topic_keywords.append(keywords_str)

        self.df["dominant_topic"] = dominant_topics
        self.df["topic_keywords"] = topic_keywords

        return self.df

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
        # Ensure train_topic_model() and assign_topics() have been called first
        self._require_model()
        self._require_topics()

        rows = []

        # Iterate over every topic ID the model knows about
        for topic_id in range(self.lda_model.num_topics):
            # Get keywords for this topic
            keywords = self.lda_model.show_topic(topic_id, topn=topn_words)
            keywords_str = ", ".join([word for word, _ in keywords])

            # Filter emails assigned to this topic
            topic_emails = self.df[self.df["dominant_topic"] == topic_id]
            num_emails = len(topic_emails)

            # Mean polarity (requires analyze_sentiment to have been called)
            mean_polarity = (
                topic_emails["polarity"].mean()
                if "polarity" in self.df.columns
                else None
            )

            # Build one row per topic and append to results list
            rows.append({
                "topic_id": topic_id,
                "keywords": keywords_str,
                "num_emails": num_emails,
                "mean_polarity": round(mean_polarity, 4)
                if mean_polarity is not None
                else None,
            })

        return pd.DataFrame(rows)

    def get_emails_by_sender(self, sender: str) -> pd.DataFrame:
        """
        Return the emails sent by a specific sender.
        """
        # Ensure load_data() has been called before this method
        self._require_dataframe()

        # Normalize input the same way load_data() data was normalized
        return self.df[self.df["sender"].str.lower() == sender.strip().lower()]

    def get_emails_by_topic(self, topic_id: int) -> pd.DataFrame:
        """
        Return the emails associated with a specific topic.
        """
        # Ensure both load_data() and assign_topics() have been called
        self._require_dataframe()
        self._require_topics()

        # Return all orws where dominant_topic matches (boolean filter)
        return self.df[self.df["dominant_topic"] == topic_id]

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
        if self.graph is None:
            raise RuntimeError("Graph not built. Call build_interaction_graph() first.")

        return {
            # Total number of unique email addresses (nodes) in network
            "num_nodes": self.graph.number_of_nodes(),
            # Total number of sender->recipient relationships (edges)
            "num_edges": self.graph.number_of_edges(),
            # Ratio of edges to maximum possible edges
            "density": round(nx.density(self.graph), 4),
            # Average number of connections per node (in + out edges)
            "avg_degree": round(
                sum(d for _, d in self.graph.degree()) / self.graph.number_of_nodes(), 4
            ),
            # 5 most connected people in network by total degree
            "top_5_by_degree": sorted(
                self.graph.degree(), key=lambda x: x[1], reverse=True
            )[:5],
        }
