## Twitter Sentiment Analysis.

#### Brand Monitoring

**Abstract**

Brand monitoring is the process of tracking different channels to identify where your brand is mentioned. Knowing where and how people are talking about your brand will help you better understand how people perceive it, and lets you collect valuable feedback from your audience. 

##### Goal

The goal of this project is' to provide a brand monitoring tool utilizing Language Processing (NLP) to derive meaningful insights, monitor market trends and to identify your target audience.

##### Data Collection

The dataset comes from CrowdFlower via data.world ([Brands and Product Emotions](https://www.crowdflower.com/data-for-everyone/)). Human raters rated the sentiment in over 9,000 Tweets as positive, negative, or neither.

##### Metrics

- Accuracy
- AUC

##### Liberary

- Numpy
- Pandas
- Seaborn
- Matplotlib
- Tensorflow
- Scikit-learn

#### Models

- TfidfVectorizer
- GloVe 

##### Conclusion

TfidfVectorizer model are able to predicts 87% of AUC accuracy despite dealing with a highly unbalanced dataset where the classes  "natural", "positive", "negative", "i can't tell" are respectively represented by 59.26%, 32.75% 6.27% and 1.72%. howevere tuning the neural network properly helped to avoid overfitting.





