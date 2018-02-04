# Yelp Research

## Equal Rating Correlation Test:

Reads the yelp data set for restaurants, users, and review ratings
`python3 restaurant_ratings_group.py`

Filters out null restaurant subgraphs.
`python3 filter_null_equal_subgraphs.py`

Run the correlation test in parallel. Need to edit file to run the correct number of :
`bash parallel_correlation_test.sh`
