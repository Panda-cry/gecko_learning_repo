import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def get_test_train_dataset():
    test = pd.read_csv('nlp_getting_started/test.csv').sample(frac=1, random_state=42)
    train = pd.read_csv('nlp_getting_started/train.csv').sample(frac=1, random_state=42)
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(train['text'].to_numpy(),
                                                                                train['target'].to_numpy(),
                                                                                shuffle=True,
                                                                                random_state=42,
                                                                                test_size=0.1)
    return train_sentences, val_sentences, train_labels, val_labels


# Pretvaranje stringova u brojeve
# Tokenizacija je one hot encoding uzima rec i daje joj vrednost 0,1,2 ect
# Moze tokenizacija da se odradi na nivou reci , slova ili sub-setova od recenice

# Postoji Embedded metoda koja za rec generise vektor nekih brojeva
# Mozemo da koritimo neki nas Embedding ili neku tuned sa tensoerflow hub


def model_1():
    train_sentences, val_sentences, train_labels, val_labels = get_test_train_dataset()
    model_1 = Pipeline([
        ("tfidf", TfidfVectorizer()),  # Konverzija text-a u broj
        ("clf", MultinomialNB())  # model za text Naive Bayas
    ])
    model_1.fit(train_sentences, train_labels)

    score = model_1.score(val_sentences, val_labels)
    predicted = model_1.predict(val_sentences)
    calculate_results(y_true=val_labels, y_pred=predicted)


def model_2():

    train_sentences, val_sentences, train_labels, val_labels = get_test_train_dataset()

    text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10000, #Maksimalna duzina vokabulara
                                      output_mode='int',
                                      output_sequence_length=15)#duzina za svaki vektor

    #Potrebno je prilagoditi recenice tj napraviti vokabular reci
    text_vectorizer.adapt(train_sentences)
    embedding = tf.keras.layers.Embedding(input_dim=10000, #duzina recnika
                              output_dim=128,
                              embeddings_initializer="uniform",
                              input_length=15)


    # inputs = tf.keras.layers.Input(shape=(1,), dtype="string")  # inputs are 1-dimensional strings
    # x = text_vectorizer(inputs)  # turn the input text into numbers
    # x = embedding(x)  # create an embedding of the numerized numbers
    # x = tf.keras.layers.GlobalAveragePooling1D()(
    #     x)  # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
    # outputs = tf.keras.layers.Dense(1, activation="sigmoid")(
    #     x)  # create the output layer, want binary outputs so use sigmoid activation
    # ml = tf.keras.Model(inputs, outputs, name="model_1_dense")

    ml = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype="string"),
        text_vectorizer,
        tf.keras.layers.Embedding(input_dim=10000,  # duzina recnika
                                  output_dim=128,
                                  embeddings_initializer="uniform",
                                  input_length=15),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    ml.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])


    ml.fit(train_sentences,train_labels,
                epochs=5,
                validation_data=(val_sentences,val_labels))


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
    -----
    y_true = true labels in the form of a 1D array
    y_pred = predicted labels in the form of a 1D array

    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    print(model_results)


def rnn():
    train_sentences, val_sentences, train_labels, val_labels = get_test_train_dataset()

    text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10000,
                                                                                   # Maksimalna duzina vokabulara
                                                                                   output_mode='int',
                                                                                   output_sequence_length=15)  # duzina za svaki vektor

    # Potrebno je prilagoditi recenice tj napraviti vokabular reci
    text_vectorizer.adapt(train_sentences)
    embedding = tf.keras.layers.Embedding(input_dim=10000,  # duzina recnika
                                          output_dim=128,
                                          embeddings_initializer="uniform",
                                          input_length=15)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,),dtype="string"),
        text_vectorizer,
        tf.keras.layers.Embedding(input_dim=10000,  # duzina recnika
                                  output_dim=128,
                                  embeddings_initializer="uniform",
                                  input_length=15),
        #tf.keras.layers.LSTM(64),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(train_sentences,train_labels,epochs=5)


def bidirection():
    train_sentences, val_sentences, train_labels, val_labels = get_test_train_dataset()

    text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10000,
                                                                                   # Maksimalna duzina vokabulara
                                                                                   output_mode='int',
                                                                                   output_sequence_length=15)  # duzina za svaki vektor

    # Potrebno je prilagoditi recenice tj napraviti vokabular reci
    text_vectorizer.adapt(train_sentences)
    embedding = tf.keras.layers.Embedding(input_dim=10000,  # duzina recnika
                                          output_dim=128,
                                          embeddings_initializer="uniform",
                                          input_length=15)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype="string"),
        text_vectorizer,
        tf.keras.layers.Embedding(input_dim=10000,  # duzina recnika
                                  output_dim=128,
                                  embeddings_initializer="uniform",
                                  input_length=15),
        # tf.keras.layers.LSTM(64),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(train_sentences,train_labels,epochs=5)

    model.evaluate(val_sentences,val_labels)


def convolution():
    train_sentences, val_sentences, train_labels, val_labels = get_test_train_dataset()

    text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10000,
                                                                                   # Maksimalna duzina vokabulara
                                                                                   output_mode='int',
                                                                                   output_sequence_length=15)  # duzina za svaki vektor

    # Potrebno je prilagoditi recenice tj napraviti vokabular reci
    text_vectorizer.adapt(train_sentences)
    embedding = tf.keras.layers.Embedding(input_dim=10000,  # duzina recnika
                                          output_dim=128,
                                          embeddings_initializer="uniform",
                                          input_length=15)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype="string"),
        text_vectorizer,
        tf.keras.layers.Embedding(input_dim=10000,  # duzina recnika
                                  output_dim=128,
                                  embeddings_initializer="uniform",
                                  input_length=15),
        # tf.keras.layers.LSTM(64),
        #tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
        tf.keras.layers.Convolution1D(filters=32, kernel_size=5, activation="relu"),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(train_sentences, train_labels, epochs=5)

    model.evaluate(val_sentences, val_labels)

#Bidirection je  sporiji ali gleda recenicu sa leva na desno i kontra  tako da ako nesto pod znake navoda ima 2 znacenje
#moze da  nauci to.
#Embedding treba da se radi za svaku posebno jer kad se jednom iskoristi ona menja strukturu vokabulara
#Sutra GRU i LSTM
#Naravno da bi trebalo ako model gresi da se uzmu par modela i da se daje preciznost prema resenju od vise modela
if __name__ == "__main__":
    convolution()
