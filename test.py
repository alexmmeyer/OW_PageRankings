import os
import pandas as pd
import variables

# for file in os.listdir(variables.RESULTS_DIRECTORY):
#     results_file_path = os.path.join(variables.RESULTS_DIRECTORY, file)
#     race_data = pd.read_csv(results_file_path)
#     print(race_data["place"][0])

# print_race_labels()
# compare_place_wr("results/2021_08_04_Tokyo_10km.csv")

# show_edges(G, "Simone Ruffini", "Andreas Waschburger")

# show_results("Alex Meyer")

# dates = ["01/31/2018", "02/28/2018", "03/31/2018", "04/30/2018", "05/31/2018", "06/30/2018", "07/31/2018", "08/31/2018",
#          "09/30/2018", "10/31/2018", "11/30/2018", "12/31/2018", "01/31/2019", "02/28/2019", "03/31/2019", "04/30/2019",
#          "05/31/2019", "06/30/2019", "07/31/2019", "08/31/2019", "09/30/2019", "10/31/2019", "11/30/2019", "12/31/2019",
#          "01/31/2020", "02/29/2020", "03/31/2020", "04/30/2020", "05/31/2020", "06/30/2020", "07/31/2020", "08/31/2020",
#          "09/30/2020", "10/31/2020", "11/30/2020", "12/31/2020", "01/31/2021", "02/28/2021", "03/31/2021", "04/30/2021",
#          "05/31/2021", "06/30/2021", "07/31/2021", "08/31/2021", "09/30/2021", "10/31/2021", "11/30/2021", "12/31/2021",
#          "01/31/2022", "02/28/2022", "03/31/2022"]
#
# dates = ["01/31/2021", "02/28/2021", "03/31/2021", "04/30/2021", "05/31/2021", "06/30/2021", "07/31/2021", "08/31/2021",
#          "09/30/2021", "10/31/2021", "11/30/2021", "12/31/2021", "01/31/2022", "02/28/2022", "03/31/2022"]
#
# for date in dates:
#     create_ranking(date, dated=True)
#     print(f"{date} ranking complete")


# race_result_file = "men/results/2018_05_20_Seychelles_10km_M.csv"
# race_data = pd.read_csv(race_result_file)
# name_list = [name.title() for name in race_data.athlete_name.tolist()]
# combos = list(combinations(name_list, 2))
# combos = [tuple(reversed(combo)) for combo in combos]
#
# for combo in combos:
#     from_name = combo[0].title()
#     to_name = combo[1].title()
#     print(f"from:{from_name} to:{to_name}")
#     # check for a tie
#     print(f"{from_name}'s place: {int(race_data.place[race_data.athlete_name == from_name])}")
#     print(f"{to_name}'s place: {int(race_data.place[race_data.athlete_name == to_name])}")
#     if(int(race_data.place[race_data.athlete_name == from_name]) == int(race_data.place[race_data.athlete_name == to_name])):
#         print("tie")
#     else:
#         print("not a tie")


# DEPRECIATION_PERIOD = 365 * 1.4
# archive_rankings_range("01/01/2017", "12/31/2017")
#
# DEPRECIATION_PERIOD = 365 * 1.7
# archive_rankings_range("01/01/2018", "12/31/2018")
#
# DEPRECIATION_PERIOD = 365 * 1.9
# archive_rankings_range("01/01/2019", "12/31/2019")
#
# DEPRECIATION_PERIOD = 365 * 1.6
# archive_rankings_range("01/01/2020", "12/31/2020")
#
# DEPRECIATION_PERIOD = 365 * 3.2
# archive_rankings_range("01/01/2021", "12/31/2021")
#
# DEPRECIATION_PERIOD = 365 * 3.6
# archive_rankings_range("01/01/2022", "04/06/2022")

list = ["a", "b", "c", "d", "e", "f"]
print(list[0:2])

for date in date_range:
    file_path = f"{gender}/rankings_archive/{alpha_date(date)}_{gender}_{RANK_DIST}km.csv"
    df = pd.read_csv(file_path)
    print(file_path)
    ranks = [int(df["rank"][df["name"] == athlete]) for athlete in athlete_list]
    horse_race_dict[date] = ranks

