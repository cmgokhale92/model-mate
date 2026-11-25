# Tool name : <u>*ModelMate*</u>

## One sentence problem statement

Business users of Team X struggle to independently generate predictions and extract predictive insights from their datasets because they lack the technical expertise required to build and execute machine learning models.

## User - Marketing Manager / Customer Retention Manager

Imagine an insurance company with 100,000 customers.

The Marketing Manager has access to data like:
- Customer demographics
- Product (policy) history
- Payment history
- Customer interactions
- Claims history

They typically want to answer:
- Which customers are likely to discontinue their policies?
- Which customers are most likely to respond to a campaign?
- What customer characteristics are associated with churn?
- Which customers should the retention team prioritize?

Traditionally, answering these questions using predictive models requires data science expertise.

## Feature 1 - AutoML

<u>Solution statement :</u> When marketing or retention managers need to understand future customer behaviour using their datasets, <u>they want an ability to train a strongest possible ML model on a click of a button</u> and <u>generate predictive insights from unseen data</u> so that they can take profitable business decisions.

### <u>Sub-Feature 1 : Automated Machine Learning (AutoML) Prediction Engine</u>

A feature that allows users to upload a structured dataset, select the variable they want to predict, and automatically train, evaluate, and execute an appropriate machine learning model to generate predictions.

Each customer will name the model they are training. Model will be tagged with the artifact, name, description, schema

How it works
1. Upload dataset: The manager uploads customer data (CSV or Excel).
2. Select prediction target: For example, select Churn as the target column.
3. Training engine : Training engine takes into account various modalities of the data, nature of the target variable, available models, cross-validation techniques and selects the best performing model.

### <u>Sub-Feature 2 : Select model and predict</u>

This feature displays the user all the models they have trained (ID, name, description, schema). User selects the model. This brings an upload button. User uploads the data in appropriate schema, model predicts and results are sent back as a downloaded file.

## Screens

The UI will have 2 screens as of now. There is no design for the screens (claude will handle that).

### Screen 1 - Data uploader

The purpose of this screen is for the user to upload 1 tabular dataset (CSV or excel). Once done, there will be multiple tabs within the screen. One tab will show the actual file with editable fields. Second tab will show the file details like total rows, file size, path, type, etc. whatever can be shown. Third tab will show the descriptive statistics of the features in numbers. Fourth tab will show the graphs and charts for each feature.

The screen will also have a form with fields like name (text box), description (text box) and a drop-down which will show all the variables in the file so that the user can select one of them as a target variable and then hit the button TRAIN MODEL.

This will trigger a BACKEND JOB 1 which does the required data cleaning, data transformation, multiple model training according to the target variable, selecting a cross-validation strategy and then finally selecting the best model as final model. Final model will be saved in DB as a row with artifact local path (joblib dump path, name, description and schema details as text - schema required for prediction). Pipeline or model will be dumped as a joblib binary.

### Screen 2 - Models

Once the model is done training, it will appear on the models page with the name, description and schema.

Models should appear like horizontal cards on the screen. The cards will have details like name, description etc.

If the user clicks on the model card, this should load the particular model at backend and then display a upload file button that takes in test data and also a sample schema required for the model.