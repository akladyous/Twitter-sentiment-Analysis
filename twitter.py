import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import seaborn as sns
import pandas as pd
import numpy as np

def make_model(metrics, input_shape, output_shape, bias_initializer, kernel_initializer, optimizer, output_bias=None):
    import keras
    import tensorflow as tf
    from keras.optimizers import Adam, RMSprop
    from keras.initializers import orthogonal, glorot_uniform
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential([
        keras.layers.Dense(units=64,activation='relu',input_dim=input_shape,
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer,
                        name='input32'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=16, activation='relu', name='dense16'),
        keras.layers.Dense(units=8, activation='relu', name='dense8'),
        keras.layers.Dense(units=output_shape,
                        activation='softmax',
                        bias_initializer=output_bias,
                        name='output4')
                        ])
    model.compile(
    optimizer=optimizer, #Adam(learning_rate=0.001)
    loss=keras.losses.categorical_crossentropy, #binary_crossentropy
    metrics=metrics
    )
    return model

def make_embedding_model(Metrics, Optimizer, Input_Dim, EmbeddinDim, Weights, Input_Length, Output_Bias=None):
    import tensorflow as tf
    from keras import Sequential, layers
    from keras.layers import GlobalMaxPool1D, LSTM, Dense, Embedding, Bidirectional, Dropout
    if Output_Bias is not None:
        Output_Bias = tf.keras.initializers.Constant(Output_Bias)

    model = Sequential()
    model.add(
        layers.Embedding(
            input_dim=Input_Dim, 
            output_dim=EmbeddinDim,
            weights=[Weights],
            input_length=Input_Length,
            trainable=False,
            mask_zero=True,
            name='embedding'
            )
        )
    model.add(Bidirectional(LSTM(units=EmbeddinDim, return_sequences=True)))
    model.add(layers.LSTM(16))
    # model.add(GlobalMaxPool1D())
    model.add(Dense(16, activation='relu', name='dense16'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax', bias_initializer=Output_Bias, name='output4'))

    model.compile(
                optimizer=Optimizer,
                loss='CategoricalCrossentropy',
                metrics=Metrics
    )
    return model


def ClassWeight(Y_true, to_array=False):
    if to_array == True:
        return np.array(Y_true.shape[0] / (np.unique(Y_true).size * np.bincount(Y_true)))
    else:
        return dict(zip( np.unique(Y_true),
                        np.array(Y_true.shape[0] / (np.unique(Y_true).size * np.bincount(Y_true))) )) 

def ohe_weights(Y_ohe):
    #  n_samples / (n_classes * np.bincount(y))
    # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    bincount = np.array([np.bincount(Y_ohe[:,x].astype(np.int))[1] for x in range(0, Y_ohe.shape[1])])
    weights = np.array(Y_ohe.shape[0] / (Y_ohe.shape[1] * bincount))
    
    ClassWeights={k:v for k,v in enumerate(weights)}

    SampleWeights = np.array([ClassWeights[np.where(row==1)[0][0]] for row in Y_ohe])
    
    return ClassWeights, SampleWeights

def CleanUp(df_series):
    import re
    # make lower case
    df_series = df_series.apply(lambda tweet: tweet.casefold() if isinstance(tweet, str) else tweet)
    # RegEx pattern : remove {link}
    link_pattern = re.compile(r"{link}")
    df_series = df_series.apply(lambda x: link_pattern.sub('', x))
    # RegEx pattern : remove RT @mention
    re_tweet_pattern = re.compile(r"(RT|retweet)((?:\b\W*@\w+)+)")
    df_series = df_series.apply(lambda x: re_tweet_pattern.sub('', x))
    # RegEx pattern : remove hashtags
    hashtag_pattern = re.compile(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)")
    df_series = df_series.apply(lambda x: hashtag_pattern.sub('', x))
    # RegEx pattern : remove RT / retweet / mention
    username_pattern = re.compile(r"(?i)((?<=\W)(RT|retweet|mention)(?=\W@?\w+))", re.IGNORECASE) # (?:@[\w_]+)
    df_series = df_series.apply(lambda x: username_pattern.sub('', x))
    # RegEx pattern : remove email address
    email_pattern = re.compile(r"[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]")
    df_series = df_series.apply(lambda x: email_pattern.sub('', x))  
    # RegEx pattern : remove HTML tags)
    html_tag_pattern = re.compile(r"<[^>\s]+>")
    df_series = df_series.apply(lambda x: html_tag_pattern.sub('', x))  
    # RegEx pattern : remove number in boundry)
    numbers_pattern = re.compile(r"\b\d+\b")
    df_series = df_series.apply(lambda x: numbers_pattern.sub('', x))  
    # RegEx pattern : remove urls
    #(https?:\/\/)(www\.)?\S+|(https?:\/\/)(www\.)\S+|(www\.)\S+
    #(\w+\.)(?:[a-zA-Z]{2,3})(?:\/)\S+
    urls_pattern = re.compile(r"(https?\:\/\/)?(www\.)?(\w+\.)?(\w+\.)([a-z]{2,3})(?=\/|\n|\s)(\S+)?")
    df_series = df_series.apply(lambda x: urls_pattern.sub('', x))
    return df_series

def tok(df_series):
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    
    tokenizer = RegexpTokenizer('[a-zA-Z0-9]+')
    df_series =  df_series.apply(lambda x: tokenizer.tokenize(x))
    
    stop_words_en = set(stopwords.words('english'))
    df_series = df_series.apply(lambda tweet: [word for word in tweet if word not in stop_words_en])
    
    lemmatizer = WordNetLemmatizer()
    df_series = df_series.apply(lambda tweet: [lemmatizer.lemmatize(word) for word in tweet])
    
    df_series = df_series.apply(lambda tweet: ' '.join(tweet))
    return df_series


def clean_stopwords(df_series):
    # Remove stopwords
    df_series = df_series.apply( lambda words: ' '.join( word.lower() for word in words.split() if word not in stop_words_en ) )
    return df_series

def hashtags(df_series):
    import re
    # return df_series.apply(lambda x: re.findall(r"#(\w+)", x.lower())).explode(ignore_index=True)
    hash_tags = dict()
    for idx, rows in df_series.iteritems():
        for hash in re.findall(r"#(\w+)", rows.lower()):
            hash_tags.setdefault(hash, 0)
            hash_tags[hash] += 1
    return dict(sorted(hash_tags.items(), key=lambda v: v[1], reverse=True))

def lemmatizer(df_series):
    lemmatized_words_dict = dict()
    for word in tweet_word_list:
        lemmatized_words_dict[word] = lemmatizer.lemmatize(word)
    return pd.DataFrame({'words': lemmatized_words_dict.keys(), 'lemmatized': lemmatized_words_dict.values()})

def Tokenization(df_series):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    lemmatized_dic = dict()
    stemmed_dic = dict()
    for idx, words in df_series.iteritems():
        if isinstance(words, str):
                for word in words.split():
                    lemmatized_word = lemmatizer.lemmatize(word)
                    if word != lemmatized_word:
                        lemmatized_dic.update({word: lemmatized_word})
                    stemmed_word = stemmer.stem(word)
                    if word != stemmed_word:
                        stemmed_dic.update({word: stemmed_word})
    df_lemmatized = pd.DataFrame({'word': lemmatized_dic.keys(), 'lemmatized': lemmatized_dic.values()})
    df_stemmed    = pd.DataFrame({'word': stemmed_dic.keys(), 'lemmatized': stemmed_dic.values()})
    return df_lemmatized, df_stemmed

def url_extractor(df_series):
    import re
    urls = set()
    urls_pattern = re.compile(r"(https?\:\/\/)?(www\.)?(\w+\.)?(\w+\.)([a-z]{2,3})(?=\/|\n|\s)(\S+)?",flags=re.IGNORECASE)
    for idx, row in df_series.iteritems():
        result = urls_pattern.search(row)
        if result:
            urls.update([result.group(0)])
    return list(urls)

class Twitter:
    def __init__(self, df_series):
        if not isinstance(df_series, pd.Series):
            raise TypeError ("Invalid data type: Pandas Series required")
        self._df_series = df_series
    
    @property
    def get_unique_words(self):
        self._unique_words = set()
        for idx, row in self._df_series.iteritems():
            if isinstance(row, str):
                for word in row.split():
                    self._unique_words.update([word])
        return list(self._unique_words)

    @property
    def WordsCount(self):
        return self._df_series.map(lambda words: len(words.split())).to_list()
    
    @property
    def get_word_list(self):
        self._words_list = list()
        for idx, row in self._df_series.iteritems():
            if isinstance(row, str):
                for word in row.split():
                    self._words_list.append(word)
        return self._words_list

    @property
    def FrequencyDist(self):
        from collections import Counter
        self._words_list = self.get_word_list
        self._word_freq_dict = dict()
        self._word_freq_dict = Counter(self._words_list)
        return dict(sorted(self._word_freq_dict.items(), key=lambda v: v[1], reverse=True))

    @property
    def ProbFrequencyDist(self):
        self._probs = dict()
        self._word_freq_dict = dict()
        self._word_freq_dict = self.FrequencyDist
        for k in self._word_freq_dict.keys():
            self._probs[k] = self._word_freq_dict[k]/sum( self._word_freq_dict.values())
        return self._probs

    def Fine_Grained_selection(self, threshold):
        """ 
        return a list of words from the vocabulary of the text that are more than X (threshold) characters long.
        so that P(w) is true if and only if w is more than X (threshold) characters long

        {w | w ∈ V & P(w)}      the set of all w such that w is an element of V (the vocabulary) and w has property P"
        """
        self._threshold = threshold
        return [word for word in self.get_unique_words if len(word)>self._threshold]

    @property
    def Lexical_Diversity(self):
        self._lex_list = list()
        for idx, row in self._df_series.iteritems():
            if isinstance(row, str):
                try:
                    self._lex_list.append(np.round(len(row) / len(set(row)),2))
                except ZeroDivisionError:
                    self._lex_list.append(0)
        return self._lex_list