from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


data = {
    'title': [
        '3 Idiots', 'Dangal', 'PK', 'Chhichhore', 'Zindagi Na Milegi Dobara',
        'Shershaah', 'Bahubali', 'Pathaan', 'War', 'Raazi',
        'Kabir Singh', 'Dear Zindagi', 'Bajrangi Bhaijaan', 'Andhadhun', 'Stree'
    ],
    'tags': [
        'college fun engineering friendship',
        'wrestling biopic motivation india',
        'alien god belief fun satire',
        'college friends motivation suicide',
        'travel friendship adventure spain',
        'army war patriotism biopic',
        'kingdom war action drama epic',
        'spy action patriotism india',
        'action thriller spy india',
        'spy thriller patriotism india',
        'love breakup emotional anger',
        'life psychology emotional therapy',
        'friendship humanity travel pakistan',
        'thriller murder piano blind',
        'horror comedy ghost village'
    ]
}

movies = pd.DataFrame(data)


cv = CountVectorizer()
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)


html = '''
<h1>🎬 Bollywood Movie Recommender</h1>
<form method="POST">
    <p>Enter Movie Name:</p>
    <input type="text" name="movie" required>
    <br><br>
    <button type="submit">Get Recommendations</button>
</form>
{% if recs %}
    <h2>Recommended Movies:</h2>
    <ul>
        {% for movie in recs %}
            <li>{{ movie }}</li>
        {% endfor %}
    </ul>
{% endif %}
'''


def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        return ["Movie not found in database."]
    idx = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]


@app.route('/', methods=['GET', 'POST'])
def index():
    recs = []
    if request.method == 'POST':
        movie = request.form['movie']
        recs = recommend(movie)
    return render_template_string(html, recs=recs)

if __name__ == '__main__':
    app.run(debug=True)

    #UMANG GHOGHARI 
