import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QListWidget


class MovieRecommender:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        print("Data loaded successfully:", self.data.shape)  # Add this line to print data shape
        self.tf_idf_matrix = self._create_tf_idf_matrix()

    def _create_tf_idf_matrix(self):
        tfidf = TfidfVectorizer(stop_words='english')
        self.data['overview'] = self.data['overview'].fillna('')
        tfidf_matrix = tfidf.fit_transform(self.data['overview'])
        return tfidf_matrix


    def recommend_movies(self, movie_title, n=5):
        # Check if the DataFrame is empty
        if self.data.empty:
            return []  # Return an empty list if DataFrame is empty

        # Find the index of the movie title
        movie_indices = self.data[self.data['title'] == movie_title].index
        if not movie_indices.empty:  # Check if movie_indices is not empty
            idx = movie_indices[0]  # Get the index of the first occurrence
            cosine_similarities = linear_kernel(self.tf_idf_matrix[idx:idx+1], self.tf_idf_matrix).flatten()
            related_movie_indices = cosine_similarities.argsort()[-(n+1):-1][::-1]
            return self.data.iloc[related_movie_indices]['title'].tolist()
        else:
            similar_titles = self._find_similar_titles(movie_title)
            if similar_titles:
                return similar_titles
            else:
                print(f"Movie '{movie_title}' not found. Please enter a valid movie title.")
                return []  # Return an empty list if movie title not found

    def _find_similar_titles(self, movie_title):
        # Convert the user input and titles to lowercase for case-insensitive comparison
        movie_title_lower = str(movie_title).lower()  # Convert to string to handle NaN values
        data_titles_lower = self.data['title'].str.lower()

        # Find exact match first
        if movie_title_lower in data_titles_lower.values:
            return []

        # Find similar movie titles based on user input
        similar_titles = []
        for title in data_titles_lower.dropna():  # Exclude NaN values
            if movie_title_lower in title or title in movie_title_lower:
                similar_titles.append(self.data.loc[data_titles_lower == title, 'title'].iloc[0])

        if similar_titles:
            print(f"Movie '{movie_title}' not found. Did you mean one of these?")
            for title in similar_titles:
                print("- " + title)
        return similar_titles

class MovieRecommenderApp(QWidget):
    def __init__(self, data_path):
        super().__init__()
        self.setWindowTitle("Movie Recommender")
        self.data_path = data_path
        self.movie_recommender = MovieRecommender(data_path)
        self.init_ui()

    def init_ui(self):
        self.movie_input = QLineEdit()
        self.recommend_button = QPushButton("Recommend")
        self.recommend_button.clicked.connect(self.show_recommendations)
        self.recommendations_list = QListWidget()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Enter a movie title:"))
        layout.addWidget(self.movie_input)
        layout.addWidget(self.recommend_button)
        layout.addWidget(self.recommendations_list)

        self.setLayout(layout)

    def show_recommendations(self):
        movie_title = self.movie_input.text()
        print("movie_title", movie_title    )
        recommendations = self.movie_recommender.recommend_movies(movie_title)
        self.recommendations_list.clear()
        self.recommendations_list.addItems(recommendations)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    data_path = 'data/movies_metadata.csv'  # Path to your dataset
    window = MovieRecommenderApp(data_path)
    window.show()
    sys.exit(app.exec_())
