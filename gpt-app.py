import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import joblib 
import os 

from flask import Flask, jsonify, request 

app = Flask(__name__)

@app.route('/', methods=['POST'])
def similarity():
    json_data = request.json  # Get JSON data from the request

    if 'user_info' in json_data:
        user_info = json_data['user_info']  # Change variable name from CV_Clear to user_info
    else:
        return jsonify({'error': 'Missing user_info in request'})

    cv = joblib.load('vect.pkl')
    df = pd.read_csv("test-jobs.csv")
    df['Combined'] = df.apply(lambda row: ' '.join([str(row['educational_qualification']), str(row['experience']),
    str(row['industry[0]']), str(row['industry[1]']),
    str(row['industry[2]']), str(row['industry[3]']),
    str(row['job_description']), str(row['job_title']),
    str(row['skills']), str(row['gender']),
    str(row['max_age']), str(row['location']), str(row['salary'])]),axis=1)
    similarity = []
    similar = []

    for i in df['Combined']:
        Match_Test = [str(user_info), str(i)]  # Adjust variable name here as well
        count_matrix = cv.transform(Match_Test)
        similarity.append(cosine_similarity(count_matrix))
        
    for i in range(0, len(similarity)):
        similar.append(similarity[i][0][1])
    
    df['Similarity'] = similar
    df_sorted = df.sort_values(by='Similarity', ascending=False)
    df_sorted = df_sorted.reset_index(drop=True)

    # Prepare output JSON data
    output_data = df_sorted.loc[0:4, ['_id', 'Similarity']].to_dict(orient='records')

    return jsonify(output_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
