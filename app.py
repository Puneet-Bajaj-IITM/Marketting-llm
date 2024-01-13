from flask import Flask, render_template, request, send_file
import os
from test import process_subsheet
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file and user input from the form
    uploaded_file = request.files['file']
    user_input = request.form['user_input']

    # Save the uploaded file
    file_path = os.path.join('uploads', uploaded_file.filename)
    uploaded_file.save(file_path)

    # Execute the search and processing function
    output_data = search_and_process(file_path, user_input)

    # Create a unique output file path
    output_file_path = os.path.join('output', 'output.xlsx')
    output_data.to_excel(output_file_path, index=False)

    # Display the processed data on the webpage
    return render_template('result.html', tables=[output_data.to_html(classes='data')], titles=output_data.columns.values)

@app.route('/download')
def download():
    # Provide a download link for the output file
    output_file_path = os.path.join('output', 'output.xlsx')
    return send_file(output_file_path, as_attachment=True)

def search_and_process(file_path, input_terms_str):
    # Split the input_terms_str into individual terms
    input_terms = input_terms_str.split(',')

    # The existing search_and_process function with slight modification
    xls = pd.ExcelFile(file_path)
    aggregated_results = pd.DataFrame()

    for term in input_terms:
        for subsheet_name in xls.sheet_names:
            df_subsheet = pd.read_excel(xls, subsheet_name)
            matching_records = df_subsheet[df_subsheet['Campaign Name'].str.contains(term, case=False, na=False)]

            if not matching_records.empty:
                processed_data = process_subsheet(subsheet_name, matching_records)
                processed_data.insert(0,'You_Searched',term)
                aggregated_results = pd.concat([aggregated_results, processed_data], ignore_index=True)


    return aggregated_results

if __name__ == '__main__':
    app.run(debug=False)
