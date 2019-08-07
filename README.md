# product-item-name-classification

For the online transactions, there are about 12K product item names in the dataset( ecommerce_product_names.csv ).

### a.) If we want to understand which catalog (such as clothing, shoes, accessories, beauty, jewelry etc.) each item is, how will you make that happen?

I requested the amazon product items with category labels.
Since one item can have multiple labels (like cuff links can be categorized as accessory and jewelry), 
it makes sense to train is a multilabel classifier with Sigmoid Cross Entropy Loss.
In this context, what I’m going to train is a multiclass classifier for simplicity.

### Machine learning

#### Workflow:

Data processing:
* Load json files and compute the labels on this Amazon product data.
* Downsample the Amazon data to make a relatively balanced classes (For speeding up the training, I chose downsampling instead of oversampling)
* Extracted the features of the product names, like TFIDF vectorizor. Now, each of 62120 product name is represented by 20761 features, representing the tf-idf score for different unigrams and bigrams.
* Dimenionality reduction. Only using the 100 best features per category.
* train test split with stratified sampling for evaluation.
* Compare the model performance between Random Forest, XGBClassifier, linearSVM, Logistic Regression, K Nearest Neighbor classifier.
* Apply the same processing process on ecommerce product names. Use all the data to train the best performing model. Get the prediction.


## Deep learning (LSTM):
### Theory explanation:
Deep learning is a set of algorithms and techniques inspired by how the human brain works. Text classification has benefited from deep learning architectures due to their potential to reach high accuracy with less need for engineered features. The main deep learning architectures used in text classification is Recurrent Neural Networks (RNN). deep-learning classifiers continue to get better the more data you feed them with.
Recurrent Neural Networks, unlike Feed-forward neural networks in which activation outputs are propagated only in one direction, the activation outputs from neurons propagate in both directions (from inputs to outputs and from outputs to inputs. This creates loops in the neural network architecture which acts as a ‘memory state’ of the neurons. This state allows the neurons an ability to remember what has been learned so far.
The vanilla recurrent neural network can work. But it suffers from two problems: vanishing gradient and exploding gradient, which make it unusable. LSTM (long short term memory) was invented to solve this issue by explicitly introducing a memory unit, called the cell into the network. The more time passes, the less likely it becomes that the next output depends on a very old input. This time dependency distance itself is as contextual information to be learned. LSTM networks manage this by learning when to remember and when to forget, through their forget gate weights.


### Workflow:
Data processing:
* Downsample to make a relatively balanced classes
* One-hot encode the labels
* Tokenize the texts then turn them into padded sequences
* Train test split the sequences and labels.

Construct NN:
* SpatialDropout1D performs variational dropout in NLP models.
* The next layer is the LSTM layer with 64 memory units.
* The output layer must create 7 output values, one for each class.
* Activation function is softmax for multi-class classification.
* Because it is a multi-class classification problem, categorical_crossentropy is used as the loss function.

Result:
After 10 epochs, on test set
Loss: 0.508
Accuracy: 0.832


<a href="url"><img src="https://github.com/JinghuiZhao/product-item-name-classification/blob/master/lstm_pred.png" align="left" height="48" width="48" ></a>
<a href="url"><img src="https://github.com/JinghuiZhao/product-item-name-classification/blob/master/lstm_pred2.png" align="left" height="48" width="48" ></a>

## Comments and thoughts:
* When it comes to creating features:
Women are the main consumers for beauty(cosmetics) and jewelery stuffs.
Men are more likely to purchase electronic devices and tools Teddoler/boys/girls are more
likely to purchase toys. If we have gender/ age group information, we can make features out of that and use
FeatureUnion from sklearn.pipeline to combine this feature with the vectorization of each
product name.
* For supervised model, since the model is trained on the labeled Amazon data, the kinds of classes should be more aglined with ecommerce product name dataset. And the data balance also matters.
* I can also try more advanced deep learning model like bi-directional LSTM.



### b.) How can you extract the additional information from the item names, such as the color, style, size, material, gender etc. if there is any?

<li> It would make sense if I extract the additional information under each class. Multileveled or hirearchical classifier would achieve better accuracy and save more time, since for each category, the adjective words can be really different. For example, in the most frequent fixed 2 grams, ‘Genuine Leather’ appears with ‘Belt’, ‘Bath’ appears with ‘Towel’, rather than ‘Genuine Leather’ appears with ‘towel’. </li>

``` CBOW and skip-gram ``` 
<li> Word2vec is one of the most popular technique to learn word embeddings using a two-layer neural network. Its input is a text corpus and its output is a set of vectors. There are two main training algorithms for word2vec, one is the continuous bag of words(CBOW), another is called skip-gram. These 2 will help us learn the similarity of words. </li>

<li> The major difference between these two methods is that CBOW is using context to predict a target word while skip-gram is using a word to predict a target context. Generally, the skip-gram method can have a better performance compared with CBOW method, for it can capture two semantics for a single word. For instance, it will have two vector representations for Apple, one for the company and another for the fruit. This is also the case in my test code. </li>

### Workflow:
* Let us focus on one category: i.e. clothing product names
* Remove the stop words and tokenize in the clothing product names
* It would be easy to come up with color word list, gender word list, size word list. After filtering the tokens in these lists, the rest will be style words.
* Genism word2vec requires that a format of ‘list of lists’ for training where every document is contained in a list and every list contains lists of tokens of that document.
* Write functions to compute the average of the word vectors in each gender/size/color/style list.
* Compute the cosine similarity between each token in a product name and each of the gender/size/color/style average vector, get the token with biggest similarity score( with a threshold sett also), this token will belong to gender/size/color/style categories.

<a href="url"><img src="https://github.com/JinghuiZhao/product-item-name-classification/blob/master/get_traits.png" align="left" height="48" width="48" ></a>
<a href="url"><img src="https://github.com/JinghuiZhao/product-item-name-classification/blob/master/attributes.png" align="left" height="48" width="48" ></a>
