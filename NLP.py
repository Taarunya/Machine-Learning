# ==========================================================
# LAB 13 – NLP USING NLTK (REAL PYTHON STYLE)
# ==========================================================

import nltk
import string
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag, FreqDist

# ----------------------------------------------------------
# DOWNLOAD REQUIRED DATA
# ----------------------------------------------------------
nltk.download('punkt')
nltk.download('punkt_tab')   
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# ----------------------------------------------------------
# SAMPLE TEXT
# ----------------------------------------------------------
text = """Chronic Kidney Disease is a serious condition that affects
the kidneys and reduces their ability to filter waste from blood."""

print("\nOriginal Text:\n", text)

# ----------------------------------------------------------
# 1. TOKENIZATION
# ----------------------------------------------------------
tokens = word_tokenize(text)
print("\nTokens:\n", tokens)

# ----------------------------------------------------------
# 2. LOWERCASE + REMOVE PUNCTUATION
# ----------------------------------------------------------
tokens = [word.lower() for word in tokens if word not in string.punctuation]

print("\nAfter Cleaning:\n", tokens)

# ----------------------------------------------------------
# 3. REMOVE STOPWORDS
# ----------------------------------------------------------
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

print("\nAfter Stopword Removal:\n", filtered_tokens)

# ----------------------------------------------------------
# 4. STEMMING
# ----------------------------------------------------------
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

print("\nStemmed Words:\n", stemmed_words)

# ----------------------------------------------------------
# 5. POS TAGGING (VERY IMPORTANT)
# ----------------------------------------------------------
pos_tags = pos_tag(filtered_tokens)

print("\nPOS Tags:\n", pos_tags)

# ----------------------------------------------------------
# 6. FREQUENCY DISTRIBUTION
# ----------------------------------------------------------
freq_dist = FreqDist(filtered_tokens)

print("\nMost Common Words:\n", freq_dist.most_common(5))

# Plot frequency
freq_dist.plot(10)
plt.title("Word Frequency Distribution")
plt.show()

# ----------------------------------------------------------
# END
# ----------------------------------------------------------
print("\n✅ NLP Lab Completed Successfully")
