from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import spacy
from fuzzywuzzy import process

app = Flask(__name__)

# Load the NLP model
nlp = spacy.load('en_core_web_sm')

# Global variable to hold the dataset
dataset = None

# NLP processing function
def nlp_processing(user_input, df):
    doc = nlp(user_input.lower())

    # Identify potential operation (e.g., mean, variance, sum)
    operation = None
    for token in doc:
        if token.lemma_ in ['mean', 'average', 'variance', 'sum', 'median', 'mode', 'standard', 'range', 'iqr', 'max', 'min', 'maximum', 'minimum', 'count', 'cumulative']:
            operation = token.lemma_

    # Extract column names from user input
    columns_in_query = []
    for chunk in doc.noun_chunks:
        column_match = process.extractOne(chunk.text, df.columns)
        if column_match and column_match[1] > 80:
            columns_in_query.append(column_match[0])

    # Fallback in case of no match
    if not columns_in_query:
        for token in doc:
            column_match = process.extractOne(token.text, df.columns)
            if column_match and column_match[1] > 80:
                columns_in_query.append(column_match[0])

    return operation, columns_in_query

# Perform statistical operations
def perform_stat_operation(operation, df, column=None):
    try:
        df = df.fillna(0)  # Handle missing values by filling with zeros

        # Use the specific column for the operation
        if column and column in df.columns:
            numeric_column = df[column].dropna()  # Filter out non-numeric values
        else:
            return f"Column '{column}' not found in the dataset."

        # Perform the requested operation
        if operation == 'mean':
            return numeric_column.mean()
        elif operation == 'median':
            return numeric_column.median()
        elif operation == 'mode':
            return numeric_column.mode()[0]
        elif operation == 'variance':
            return numeric_column.var()
        elif operation == 'standard':
            return numeric_column.std()
        elif operation == 'range':
            return numeric_column.max() - numeric_column.min()
        elif operation == 'iqr':
            return numeric_column.quantile(0.75) - numeric_column.quantile(0.25)
        elif operation == 'sum':
            return numeric_column.sum()
        elif operation == 'count':
            return numeric_column.count()
        elif operation == 'cumulative':
            return numeric_column.cumsum().to_list()
        elif operation == 'max':
            return numeric_column.max()
        elif operation == 'min':
            return numeric_column.min()
        else:
            return f"Operation '{operation}' not recognized."
    except Exception as e:
        return f"Error performing operation: {e}"

@app.route('/', methods=['GET', 'POST'])
def home():
    global dataset
    result = None

    if request.method == 'POST':
        query = request.form['query']

        # Process the query if dataset is loaded
        if dataset is not None:
            operation, columns_in_query = nlp_processing(query, dataset)

            if operation and columns_in_query:
                column_name = columns_in_query[0]
                result = perform_stat_operation(operation, dataset, column=column_name)
            else:
                result = "Couldn't interpret your query."

        return jsonify({"query": query, "result": result})

    return render_template('index.html', dataset=dataset.head(5).to_html(index=False) if dataset is not None and not dataset.empty else None)

# Dataset upload route
@app.route('/upload', methods=['POST'])
def upload():
    global dataset
    file = request.files['file']
    if file:
        dataset = pd.read_csv(file, sep='\t', header=None)
        return redirect(url_for('column_names'))
    return redirect(url_for('home'))

# Column names input route
@app.route('/columns', methods=['GET', 'POST'])
def column_names():
    global dataset

    if request.method == 'POST':
        column_names = request.form.getlist('column_name')
        if dataset is not None:
            dataset.columns = column_names  # Set the column names
        return redirect(url_for('home'))

    # Pass the actual DataFrame (not converted to HTML) for column name entry
    return render_template('columns.html', dataset=dataset)

if __name__ == '__main__':
    app.run(debug=True)
