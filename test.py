import os
import pandas as pd
import os
import re
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
import re
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from io import StringIO

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# # Load spaCy model
# spacy_model_path = 'model\en_core_web_sm-3.7.1\en_core_web_sm\en_core_web_sm-3.7.1'
# nlp = spacy.load(spacy_model_path)

import spacy
import inflect
import inflect
import spacy
import inflect


import re
import spacy
import inflect

nlp = spacy.load("en_core_web_sm")
import re
import spacy
import inflect


def preprocess_search_term(search_term):
    # Remove symbols, convert to lowercase, and strip leading/trailing spaces
    search_term = search_term.strip().lower()
    search_term = re.sub(r'[^a-zA-Z\s]', '', search_term)

    doc = nlp(search_term)
    lemmas = []

    p = inflect.engine()

    for token in doc:
        # Check if the term is a noun, verb, adjective, or adverb
        if token.pos_ in {'NOUN', 'VERB'}:
            # Use spaCy's lemmatization for verbs
            lemma = token.lemma_
            # Convert plural nouns to singular
            if token.pos_ == 'NOUN' and p.singular_noun(lemma):
                lemma = p.singular_noun(lemma)
            lemmas.append(lemma[:-3] if lemma.endswith("ing") else lemma)

    result = set()

    for i in range(len(lemmas)):
        # Phrases of length 1
        result.add(lemmas[i])
        if i < len(lemmas) - 1:
            # Phrases of length 2
            result.add(f"{lemmas[i]} {lemmas[i + 1]}")
            if i < len(lemmas) - 2:
                # Phrases of length 3
                result.add(f"{lemmas[i]} {lemmas[i + 1]} {lemmas[i + 2]}")

    return result

# Example usage:
search_term = "Cutting and from an and the then hardly harding gardening tools"
result = preprocess_search_term(search_term)
print(result)

vectorizer = TfidfVectorizer()
kmeans = KMeans(n_clusters=3, random_state=42)


def process_subsheet(subsheet_name, df_subsheet):
    # Remove records with blank or single-space customer search terms
    df_subsheet = df_subsheet[df_subsheet['Customer Search Term'].str.strip() != '']

    if df_subsheet.empty:
        # No valid records left after filtering, skip processing
        return pd.DataFrame()

    df_subsheet['processed_search_terms'] = df_subsheet['Customer Search Term'].apply(preprocess_search_term)

    # Drop rows with NaN in 'processed_search_terms' column
    df_subsheet = df_subsheet.dropna(subset=['processed_search_terms'])

    if df_subsheet.empty:
        # No valid records left after dropping NaN, skip processing
        return pd.DataFrame()

    df_processed = df_subsheet.explode('processed_search_terms')

    # Check for NaN values in 'processed_search_terms' column
    df_processed = df_processed.dropna(subset=['processed_search_terms'])

    if df_processed.empty:
        # No valid records left after dropping NaN, skip processing
        return pd.DataFrame()

    X = vectorizer.fit_transform(df_processed['processed_search_terms'])  # Fit and transform

    df_processed['cluster_label'] = kmeans.fit_predict(X)  # Fit and predict

    # Create 'Frequency' column indicating the count of each term
    df_processed['Frequency'] = 1

    # Aggregate metrics by cluster
    cols_to_agg = [
        '7 Day Total Sales (₹)', '7 Day Total Orders (#)',
        '14 Day Total Sales (₹)', '14 Day Total Orders (#)'
    ]

    existing_cols = [col for col in cols_to_agg if col in df_processed.columns]
    existing_cols.extend(['Impressions', 'Clicks', 'Spend'])

    if not existing_cols:
        raise KeyError(f"Column(s) {cols_to_agg} do not exist")

    # Aggregate data to calculate 'Frequency' as the number of times a phrase appears
    aggregated_data = df_processed.groupby(['processed_search_terms']).agg(
        {col: 'sum' for col in existing_cols},
        Frequency=('processed_search_terms', 'count')  # Count the occurrences
    ).reset_index()

    # Ensure '7 Day' and '14 Day' columns are present in the DataFrame
    for day_col in ['7 Day Total Sales (₹)', '7 Day Total Orders (#)', '14 Day Total Sales (₹)', '14 Day Total Orders (#)']:
        if day_col not in existing_cols:
            aggregated_data[day_col] = 0

    # Fill missing values in the aggregated_data DataFrame with 0
    aggregated_data.fillna(0, inplace=True)

    # Add 'Frequency' column indicating the count of each term
    aggregated_data['Frequency'] = df_processed.groupby('processed_search_terms')['Frequency'].sum().reset_index()['Frequency']

    # Aggregate 7 Day and 14 Day metrics, treating missing values as 0
    aggregated_data['7 Day Total Sales (₹)'] = aggregated_data['7 Day Total Sales (₹)'].fillna(0)
    aggregated_data['7 Day Total Orders (#)'] = aggregated_data['7 Day Total Orders (#)'].fillna(0)
    aggregated_data['14 Day Total Sales (₹)'] = aggregated_data['14 Day Total Sales (₹)'].fillna(0)
    aggregated_data['14 Day Total Orders (#)'] = aggregated_data['14 Day Total Orders (#)'].fillna(0)

    # Calculate additional metrics
    aggregated_data['Total Sales (₹)'] = aggregated_data['7 Day Total Sales (₹)'] + aggregated_data['14 Day Total Sales (₹)']
    aggregated_data['Total Orders (#)'] = aggregated_data['7 Day Total Orders (#)'] + aggregated_data['14 Day Total Orders (#)']
    aggregated_data['Click-Thru Rate (CTR)'] = (aggregated_data['Clicks'] / aggregated_data['Impressions']).map(lambda x: f'{x:.2%}' if not pd.isna(x) else '')
    total_sales_col = 'Total Sales (₹)'
    # Check if total sales are greater than zero before calculating ACOS
    aggregated_data['ACOS'] = (aggregated_data['Spend'] / aggregated_data[total_sales_col]).map(
        lambda x: f'{x:.2%}' if x > 0 and not pd.isna(x) else '0.00%'
)
    aggregated_data['Conversion rate'] = (aggregated_data[existing_cols[1]] / aggregated_data['Clicks']).map(lambda x: f'{x:.2%}' if not pd.isna(x) else '')

    # Reorder columns for clarity
    aggregated_data = aggregated_data[['processed_search_terms', 'Frequency',
                                       'Total Sales (₹)', 'Total Orders (#)',
                                       'Impressions', 'Clicks', 'Spend', 'Click-Thru Rate (CTR)',
                                       'ACOS', 'Conversion rate']]

    return aggregated_data

def search_and_process(input_folder, output_folder, input_terms):

    # Rest of the function...

    # Function to search and process data based on input terms
    aggregated_results = pd.DataFrame()

    for term in input_terms:
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".xlsx"):
                input_file_path = os.path.join(input_folder, file_name)
                xls = pd.ExcelFile(input_file_path)

                for subsheet_name in xls.sheet_names:
                    df_subsheet = pd.read_excel(xls, subsheet_name)
                    matching_records = df_subsheet[df_subsheet['Campaign Name'].str.contains(term, case=False, na=False)]

                    if not matching_records.empty:
                        processed_data = process_subsheet(subsheet_name, matching_records)
                        processed_data['You_Searched'] = term
                        aggregated_results = pd.concat([aggregated_results, processed_data], ignore_index=True)

    return aggregated_results

def main():
    # Main function to orchestrate the execution
    input_folder = "input"
    output_folder = "output"

    # Take user input for search terms
    user_input = input("Enter search terms separated by commas: ")
    search_terms = [term.strip() for term in user_input.split(',')]

    # Execute the search and processing function
    output_data = search_and_process(input_folder, output_folder, search_terms)

    # Save the aggregated results into a single output file
    output_file_path = os.path.join(output_folder, "output.xlsx")
    output_data.to_excel(output_file_path, index=False)

    # Display a message when the processing is complete
    print(f"Output saved to the 'output' folder.")

if __name__ == "__main__":
    main()
