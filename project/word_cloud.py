import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/input/train.csv')
data = data[data['toxic'] == 1]
# Combine all comments into one giant string
text = " ".join(comment for comment in data.comment_text)

# Add any additional stopwords
stopwords = set(STOPWORDS)

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# Save the image in the img folder:
plt.savefig('wordcloud.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
