# Depression_detection_usin_CNNandLSTM
In this project the tweets are extracted manually using tweets.py and collected over 20k+ tweets from twitter using different keyword into one CSV file (fina_output.csv).
Tweets till 5000 are labelled manually and used as a dataset for this project.

Procedure to run through the whole project:

1. Run tweets.py to collect tweet from the twitter.
2. label them as depressed or not (can also use the database i used)
3. Run the main.py file to run the model

# model.py contains the architecture of the model used
# reports are also attached as which algorithm worked with how much accuracy 
# Decision tree worked better comapring to other machine learning algo, with the accuracy of nearly 90%
# CNN + LSTM outperformed and had an average of 96% accuracy
