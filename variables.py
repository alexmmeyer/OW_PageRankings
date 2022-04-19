import pandas

gender = "men"

event_points = pandas.read_csv("event_points.csv")
athlete_countries = pandas.read_csv("athlete_countries.csv")
RESULTS_DIRECTORY = gender + "/results"
RANKING_FILE_NAME = "PageRanking.csv"

# Depreciation Period: time period in days over which a depreciation is applied to the initial weight of a result.
DEPRECIATION_PERIOD = 365 * 3.6
DEPRECIATION_MODEL = "linear"
# Drives age_weight_exp() exponential decay function. The more negative, the quicker the decline in age_weight.
LAMBDA = -1.4

# Tune ranking for the following distance in km. 0 for no preference, or best "overall" regardless of dist.
RANK_DIST = 10
