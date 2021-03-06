from scikits.crab import datasets
from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.recommenders.knn import UserBasedRecommender
from scikits.crab.similarities import UserSimilarity


movies = datasets.load_sample_movies()
songs = datasets.load_sample_songs()

print movies.data
print movies.user_ids
print movies.item_ids

#Build the model
model = MatrixPreferenceDataModel(movies.data)

#Build the similarity
similarity = UserSimilarity(model, pearson_correlation)

#Build the User based recommender_sample.py
recommender = UserBasedRecommender(model, similarity, with_preference=True)

#Recommend items for the user 5 (Toby)
print recommender.recommend(5)


