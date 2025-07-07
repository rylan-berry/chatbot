This is my model, it is based on my original LLM which I made months ago, along with a more recently made tokenizer. The model is broken across multiple files so preprocesses can be run seperatly from model training, as well as allowing the model to be run seperately from training as well. The model should be abel to run right out of the box as long as the generate.py, model.py, vocabulary_aid.py, model.pt, and merges.json are installed and working correctly.


There are external librarys necesary to run the model. The only one necesary to run the model is PyTorch. However, if you wish to train the model, NumPy is required. And if you want to adjust the pre-training processes, specifically the data collection, BeautifulSoup is needed.


If you are completly training the model from scratch, here's the order of which the files need to be run. First run trainVocab.py, this sets up the vocabulary to be used for encoding and decoding. Then, run dataCollect.py, this gathers all the wanted sites that are wanted to be used in training into a single encoded file. Currently, all data is gathered from www.gutenberg.org. With these processes complete, run train.py, this will train the model off of the data. Now, once trained, run generate.py to test the model.


Plans for the model:  Implement a fine-tuning system to turn it into a chatbot; set up messaging with the bot.
