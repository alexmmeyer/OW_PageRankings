import pandas

event_points = pandas.read_csv("event_points.csv")
RESULTS_DIRECTORY = "results"
RANKING_FILE_NAME = "PageRanking.csv"

# Must be in MM/DD/YYYY format:
RANKING_AS_OF = "08/20/2018"

# Depreciation Period: time period in days over which a depreciation is applied to the initial weight of a result.
DEPRECIATION_PERIOD = 365 * 4
DEPRECIATION_MODEL = "exponential"

# Drives age_weight_exp() exponential decay function. The more negative, the quicker the decline in age_weight.
LAMBDA = -1.4

# Tune ranking for the following distance in km. 0 for no preference, or best "overall" regardless of dist.
RANK_DIST = 11
