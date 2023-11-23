# Import all modules
import pandas as pd, numpy as np, streamlit as st, pickle
from sklearn.feature_extraction import FeatureHasher
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error

# Load the trained model
@st.cache_data()
def load_model():
    pickle_in = open("Models/linear_regression.pkl", "rb")
    model = pickle.load(pickle_in)
    pickle_in.close()
    return model

# Load data and models
df = pd.read_csv("amazon_data.csv")
df = df.drop(columns=["asin", "imgUrl", "productURL"],axis=1)
df = df[df["reviews"] > 0]

## Pre-Processing
### Shuffle rows
# Rows have to be shuffled otherwise there will be issues due the test-train split.
# Data is currently sorted by categoryName, this will cause problems later unless shuffled before modelling.
index_array = np.arange(len(df))
np.random.shuffle(index_array)

df = df.iloc[index_array].reset_index(drop=True)

### isBestSeller: bool -> int
dict_map = {True: 1, False: 0}
df['isBestSeller'] = df['isBestSeller'].map(dict_map)


### categoryName: string -> float  -- FeatureHasher
n_features = len(df.categoryName.unique())
categories = df.categoryName.astype(str)  
categories = [[category] for category in categories]
hasher = FeatureHasher(n_features=n_features, input_type="string")
X_category = hasher.transform(categories).toarray().astype("float32")
hashed_df = pd.DataFrame(X_category, columns=[f"hash_{i}" for i in range(n_features)])

# Concatenate dataframes
data = pd.concat([df, hashed_df], axis=1)
data = data.drop(axis=1,columns="categoryName")


### title: string -> int -- Word2Vec
model_path = "word2vec_model.model"
model_w2v = Word2Vec.load(model_path)

# Define a function to get the word vectors for the first 5 words of a product name
def get_word_vectors(product_name):
    try:
        five_words = word_tokenize(product_name.lower())[:5]
        vectors = [model_w2v.wv[word] for word in five_words]
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.nan
    except KeyError:
        return np.nan      # Handle the case where a word is not in the vocabulary

data['average_vector'] = data['title'].apply(get_word_vectors)

# Expand the average vector into several columns
for i in range(model_w2v.vector_size):
    data[f'embedding_dim_{i + 1}'] = data['average_vector'].apply(lambda x: x[i] if isinstance(x, np.ndarray) else np.nan)

numerical = data.drop(axis=1, columns=["title", "average_vector"])

## Data Split and Transformation
y = numerical.price.values
X = numerical

X_train = X[0:-40000].drop(axis=1,columns="price").values
y_train = y[0:-40000]
y_train_transformed = np.log1p(y_train)

X_test = X[-40000:].drop(axis=1,columns="price").values
y_test = y[-40000:]


## Modelling
### LinearRegression()
LR = LinearRegression()
LR.fit(X_train, y_train_transformed)
y_pred_LR = LR.predict(X_test)
y_pred_LR = np.expm1(y_pred_LR)

### ElasticNet()
EN = ElasticNet()
EN.fit(X_train, y_train)
y_pred_EN = EN.predict(X_test)

### Comparison
compare = pd.DataFrame({"productName": df.title.iloc[-40000:],
                        "rating": df.stars.iloc[-40000:],
                        "Reviews": df.reviews.iloc[-40000:],
                        "actualPrice": y_test,
                        "y_pred_LR": y_pred_LR,
                        "y_pred_EN": y_pred_EN
                        }).reset_index()

print(compare)


## Evaluation
mse = mean_squared_error(y_test, y_pred_LR)
print(f'Mean Squared Error LinearRegression: {mse}')
mse = mean_squared_error(y_test, y_pred_EN)
print(f'Mean Squared Error ElasticNet: {mse}')