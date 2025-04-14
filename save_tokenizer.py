import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load your training data
df = pd.read_csv("C:/Users/funnn/Desktop/AI SEM 6 project/spam-detector/SpamTrain.csv", encoding='latin-1')

# Extract messages column (v2)
texts = df['v2'].astype(str).tolist()

# Create and fit tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Save tokenizer to a .pickle file
with open("C:/Users/funnn/Desktop/AI SEM 6 project/spam-detector/tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Tokenizer saved successfully at tokenizer.pickle")
