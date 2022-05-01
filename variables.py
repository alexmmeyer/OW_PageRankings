import pandas

GENDER = "women"

event_weights = pandas.read_csv("event_points.csv")
athlete_countries = pandas.read_csv("athlete_countries.csv")
RESULTS_DIRECTORY = GENDER + "/results"
RANKINGS_DIRECTORY = GENDER + "/rankings_archive"
RANKING_FILE_NAME = "PageRanking.csv"

# Depreciation Period: time period in days over which a depreciation is applied to the initial weight of a result.
DEPRECIATION_PERIOD = 365 * 3.65
DEPRECIATION_MODEL = "linear"
# Drives age_weight_exp() exponential decay function. The more negative, the quicker the decline in age_weight.
LAMBDA = -1.4

# Tune ranking for the following distance in km. 0 for no preference, or best "overall" regardless of dist.
RANK_DIST = 10

# Predictability settings: optimize test for a subset of the overall ranking.
FROM_RANK = 1
TO_RANK = 100
