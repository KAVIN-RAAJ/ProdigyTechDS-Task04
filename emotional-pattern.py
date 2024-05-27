from textblob import TextBlob
import matplotlib.pyplot as plt

# Sample text data
text_data = [
    "I love this product! It's amazing!",
    "The service was terrible. I will never come back again.",
    "Today is a beautiful day.",
    "I'm feeling neutral about this.",
    "This movie was so boring.",
    "I'm very happy with the outcome.",
    "The food was delicious!"
]

# Calculate sentiment scores
sentiment_scores = [TextBlob(text).sentiment.polarity for text in text_data]

# Create histogram
n, bins, patches = plt.hist(sentiment_scores, bins=20, color='skyblue', edgecolor='black')

# Annotate the histogram bins with their values
for count, x in zip(n, bins[:-1]):
    if count > 0:  # Only annotate bins with counts
        plt.text(x + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom')

# Plot info
mean_score = sum(sentiment_scores) / len(sentiment_scores)
min_score = min(sentiment_scores)
max_score = max(sentiment_scores)

# Display text information on the plot
textstr = '\n'.join((
    f'Mean Sentiment: {mean_score:.2f}',
    f'Min Sentiment: {min_score:.2f}',
    f'Max Sentiment: {max_score:.2f}',
    f'Total Samples: {len(sentiment_scores)}'))

# Place text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
