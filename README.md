# Pseudo Code for the Speech Recognition Model

1.	Converting the audio (.wav) files into texts and saving the content in .csv format,
a.	With the help of ‘speech_recognition’ module in python each audio files are converted into text/sentences,
b.	The generated sentences are kept in an array format,
c.	Each such sentences are labelled with the language (e.g. ‘Karan applicad padhna hi’ is labelled with ‘marathi’),
d.	Hence the array contains ‘Filename’, ‘Text’ and ‘language’ columns where, ‘Filename’ contains the name of the audio file, ‘Text’ contains the converted audio text and the ‘language’ column contains the label of the generated text,
e.	The array is saved/downloaded in .csv (comma separated)  format,
f.	This process is followed for every language data.

2.	Uploading the generated files (.csv) as python dataframe
a.	Specifying the path where the .csv files reside,
b.	Appending the data in a series format,
c.	Preparing the datframe (having 'Filename','text','language' columns),

3.	Converting the ‘language’ column of the dataframe into numbers
a.	Preparing an array (dictionary) containing the mapping between the language and numbers,
b.	Replacing the language values with numbers w.r.t the defined array,

4.	Converting the ‘Text’ column of the dataframe into matrix
a.	With the help of ‘TfidfVectorizer’ module of python the ‘text’ column is converted into a numeric array,
b.	This conversion is based on certain logic of ‘Term frequency X Inverse document Frequency’
c.	Again, converting the array to dataframe format for training and testing preparation.

5.	Training and Testing split of data
a.	Separating the 0.2 part of the total data (in dataframe format) for the testing purpose,
b.	Remaining data will be used for training the model, 

6.	Transfer learning (it is the reuse of a pre-trained model on a new problem) a Sequential model with required perimeters
a.	Sequential means linear stack of layers. We can create a Sequential model by passing a list of layer instances to the constructor,
b.	Adding an input layer (with 100 input data and relu output), an output layer (with 7 outputs and softmaz output) and a hidden layer (with relu output and same dimension with the input layer),
c.	Defining a Stochastic gradient descent optimizer for optimizing the algorithm performance,
d.	Then compiled the model with ‘categorical_crossentropy ‘as loss function and ‘accuracy’ as matrix,
e.	After that the training data is used to train the model with fit function,
f.	Batch size (batch size is a number of samples processed before the model is updated) is selected as 10, and the epoch (The number of epochs is the number of complete passes through the training dataset) is kept as 20.
g.	Validation split is kept as 0.1 to use 1 data to verify the accuracy in each epoch run,

7.	Prediction from the model:
a.	Then testing data is provided to the model and the predicted data is collected,
b.	The predicted data and the Test data is compared to formulate the actual accuracy of the model,
c.	Also have created a confusion matrix based on the predicted values,
d.	The input to the confusion matrix is utilized to find the precision, recall and f1 score of the model.

8.	Live prediction with the model
a.	The device microphone is used for the audio input,
b.	With the help of ‘speech_recognition’ module the input audio is converted into text,
c.	The text is then converted into array,
d.	The array is fed as an input to the model,
e.	The output from the model is converted into language string, with the help of earlier defined dictionary,
f.	The said text and the output is shown as a result in runtime.


Links to refer:
•	https://ruder.io/optimizing-gradient-descent/
•	https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/#:~:text=The%20batch%20size%20is%20a%20number%20of%20samples%20processed%20before,samples%20in%20the%20training%20dataset.
