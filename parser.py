import json
import logging
import sys

from pprint import pprint
from scikits.crab.metrics import pearson_correlation
from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.recommenders.knn import UserBasedRecommender
from scikits.crab.similarities import UserSimilarity


#with open('yelp_training_set/yelp_training_set_business.json') as f:
#    businesses = [json.loads(line) for line in f]

#print businesses[0]["name"]

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


with open('yelp_training_set/yelp_training_set_review.json') as f:
    reviews = [json.loads(line) for line in f]

with open('yelp_test_set/yelp_test_set_review.json') as f:
    reviews_test = [json.loads(line) for line in f]

#print reviews[0]

users_rev_dic = {}

for rev in reviews:
    if not users_rev_dic.has_key(rev["user_id"]):
        biz_rev_dic = {rev["business_id"]: rev["stars"]}
        users_rev_dic[rev["user_id"]] = biz_rev_dic
    else:
        users_rev_dic[rev["user_id"]][rev["business_id"]] = rev["stars"]

print 'dic built'

#pprint(users_rev_dic)

#Build the model
model = MatrixPreferenceDataModel(users_rev_dic)

print 'Model built'

#Build the similarity
similarity = UserSimilarity(model, pearson_correlation)

print 'Similarity built'

#Build the User based recommender
recommender = UserBasedRecommender(model, similarity, with_preference=True)

print 'Recommender built'

recommender.estimate_preference("EMpFiVyiaMS58XsLZdS6vA", 'QL3vFMAsEHqfi1KGH-4igg')

print 'end'




