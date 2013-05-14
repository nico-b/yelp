import json
import logging
import sys

from scikits.crab.metrics import pearson_correlation
from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.recommenders.knn import UserBasedRecommender
from scikits.crab.similarities import UserSimilarity


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


with open('yelp_training_set/yelp_training_set_review.json') as f:
    reviews = [json.loads(line) for line in f]

with open('yelp_test_set/yelp_test_set_review.json') as f:
    reviews_test = [json.loads(line) for line in f]

users_rev_dic = {}

for rev in reviews:
    if not users_rev_dic.has_key(rev["user_id"]):
        biz_rev_dic = {rev["business_id"]: rev["stars"]}
        users_rev_dic[rev["user_id"]] = biz_rev_dic
    else:
        users_rev_dic[rev["user_id"]][rev["business_id"]] = rev["stars"]

print 'dic built'

#Build the model
model = MatrixPreferenceDataModel(users_rev_dic)

print 'Model built'

#Build the similarity
similarity = UserSimilarity(model, pearson_correlation)

print 'Similarity built'

#Build the User based recommender.py
recommender = UserBasedRecommender(model, similarity, with_preference=True)

print 'Recommender built'

recommender.estimate_preference("EMpFiVyiaMS58XsLZdS6vA", 'QL3vFMAsEHqfi1KGH-4igg')

print 'end'