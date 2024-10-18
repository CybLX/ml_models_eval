# ML Models Evaluation on Comment Datasets

## Overview
This project evaluates various machine learning models on comment datasets from several sources including B2W, Buscape, Olist, Tweet, and UTLCORPUS. The focus is on text classification tasks using different methods such as `tfidf` and `count` vectorization, and applying various machine learning models to analyze performance metrics.

## Dataset and Preprocessing

**IMPORTANT:** Due to the dataset being in Portuguese and consisting of 1.6 GB of comments, it was compressed into a .zip file along with preprocessing/cleaning scripts, i will set up Git Large File Storage (LFS) soon. Below is a description of each dataset used and their links. Feel free to download them and run the preprocessing scripts:

### B2W:
- **Description**: B2W-Review01 contains over 130,000 customer reviews collected from Americanas.com between January and May 2018.
- **Source**: [B2W Reviews01](https://github.com/americanas-tech/b2w-reviews01)

### Buscapé:
- **Description**: A dataset with over 80,000 product reviews tracked in 2013.
- **Source**: [Buscapé Reviews](https://drive.google.com/file/d/1IZJuvt1uxQ4oPGAvGQQxQ_h_ZiV-Be72/view)

### OLIST:
- **Description**: A dataset containing 100,000 orders from 2016 to 2018 across various marketplaces.
- **Source**: [Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_order_items_dataset.csv)

### Twitter:
- **Description**: A dataset of 890,000 tweets, both topic-based and general, collected between August and October 2018.
- **Source**: [Portuguese Tweets for Sentiment Analysis](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis)

### UTL Corpus:
- **Description**: A corpus with 2,881,589 reviews on movies and smartphone apps.
- **Source**: [UTL Corpus](https://github.com/RogerFig/UTLCorpus)

## Dataset Features
The datasets contain customer reviews, tweets, and other user-generated content, and they have been preprocessed to include several features:
- **Tokens (stemmed and lemmatized):** Processed words from the original text.
- **POS Tags:** Part-of-speech tagging to identify the syntactic roles of words.
- **Syntactic Dependency:** Information about how words are related to each other in the sentence.
- **Polarity:** Sentiment analysis to classify whether the comments are positive or negative.

## Project Goals
The primary goal of this project is to:
1. **Preprocess text data**: Remove outliers, handle stopwords, and apply different vectorization methods.
2. **Build a robust vocabulary**: Filter words based on frequency and apply n-gram models.
3. **Evaluate machine learning models**: Test various classifiers such as Naive Bayes, Logistic Regression, SVM, Random Forest, and Gradient Boosting on these datasets.
4. **Generate performance reports**: Evaluate models based on metrics like accuracy, F1 score, AUC, kappa, and more.

## Tools Used
- **Libraries:**
  - `scikit-learn`: For machine learning models and data manipulation.
  - `pandas`: Data handling and preprocessing.
  - `argparse`: For managing command-line arguments.
  - `LaTeX`: To generate performance reports in PDF format.
  - `PyTorch`: For efficient dataset handling and splitting.

- **Models Evaluated:**
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
  - k-Nearest Neighbors (KNN)
  - Random Forest
  - Stochastic Gradient Descent (SGD)
  - Gradient Boosting

## How to Use
1. **Clone the Repository**:
   `git clone https://github.com/cyblx/ml_models_eval`

2. **Install Dependencies**:
   Run `pip install -r requirements.txt` to install the necessary libraries.

3. **Prepare Dataset**:
   Place your datasets in the `./dataset/` directory. Ensure that the comments are formatted according to the project specifications or unzip the proposed dataset.

4. **Run the Training**:
   Use the following command to start training:
   ```bash
   python main.py --help
   ```
   Adjust parameters according to your needs using the argparse options.

5. **Generate Performance Reports**:
   After training, performance metrics are saved in a JSON file, and a LaTeX-generated PDF report will be created in the `./report` directory.

## For More Information
For more information, codes, tutorials, and exciting projects, visit the links below:

- Email: alves_lucasoliveira@usp.br
- GitHub: [cyblx](https://github.com/cyblx)
- LinkedIn: [Cyblx](https://www.linkedin.com/in/cyblx)
