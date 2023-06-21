# Lyrics Genre Classification
The model takes in a sample of approximately 1,000 well-known songs per genre, analyzes the lyrics, and predicts the probability of each genre for a given set of lyrics.

## How to Use the Application
1. Navigate to the following page: https://team12lyricprediction-bbfd6d8898af.herokuapp.com/#
2. The first page provides a general overview of the creation process behind the app. To being making predictions, click on _Try the App Now_ (in the black box) or click on the _App_ tab above, or click on this link (https://team12lyricprediction-bbfd6d8898af.herokuapp.com/prediction.html).
3. Input any set of lyrics or text in the text box on the page.
4. Click submit.


## Repository Contents:
_Jupyter Notebooks_:

There are five jupyter notebooks, one for each major step in the final product development.
1. [_API_DataPull_Jupyter Notebook_](https://github.com/hunterblum/TextMining_Team12/blob/main/01_API_DataPull.ipynb)
    - The first notebook contains the code for pulling raw lyrical data from the Genius API.
2. [_Preprocessing_Jupyter Notebook_](https://github.com/hunterblum/TextMining_Team12/blob/main/02_PreProcessing.ipynb)
    - The second notebook contains the code for preprocessing the raw data to acquire normalized clean text, as well as tokens.
3. [_EDA_Jupyter Notebook_](https://github.com/hunterblum/TextMining_Team12/blob/main/03_EDA.ipynb)
    - The third notebook contains the code for general exploratory data analysis, where aspects such as genre distribution, song length, and other descriptive statistics about the songs pulled were explored.
4. [_Modeling_Jupyter Notebook_](https://github.com/hunterblum/TextMining_Team12/blob/main/04_Modeling.ipynb)
    - The fourth notebook contains the code for establishing machine learning models in the context of a multilabel problem. This is the foundation for the underlying model that powers the final application. Ultimately, a linear SVC model was selected as the optimal model.
5. [_ModelExplanation_Jupyter Notebook_](https://github.com/hunterblum/TextMining_Team12/blob/main/05_ModelExplanation.ipynb)
    - The fifth notebook contains the code for examining how the model predicted the likelihood of each genre per lyric set via eli5 analysis.

_Lyric Data_

- [_Raw_Genius_API_Data_](https://github.com/hunterblum/TextMining_Team12/blob/main/data/lyrics.csv.gz)
    - The raw data set contains the data pulled directly from the Genius API, without any preprocessing or other alterations.

- [_Preprocessed_Data_](https://github.com/hunterblum/TextMining_Team12/blob/main/data/genre_prepped.csv.gz)
    - The preprocessed data contains results following the completed run of the second notebook. The data here is ready for modeling.

_Flask Application Files_:

- _Templates Folder_(https://github.com/hunterblum/TextMining_Team12/tree/main/templates)
    - This folder contains the files needed to create the foundation for the model, such as all HTML source code, the model and text vectorizer pickle objects, and the python script utilized for predictions.
- _Static Folder_(https://github.com/hunterblum/TextMining_Team12/tree/main/static)
    -  This folder contains the CSS files needed by Flask for aesthetics and application design.
-  _App.py_(https://github.com/hunterblum/TextMining_Team12/blob/main/app.py)
    -   App.py is the file read by Flask that drives the application.
