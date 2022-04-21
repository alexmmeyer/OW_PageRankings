import pandas as pd
import os
from datetime import datetime as dt
from datetime import timedelta
from datetime import date
from itertools import combinations
import variables
import networkx as nx
import time
import matplotlib.pyplot as plt
import math
import seaborn as sb

RESULTS_DIRECTORY = variables.RESULTS_DIRECTORY
RANKING_FILE_NAME = variables.RANKING_FILE_NAME
DEPRECIATION_PERIOD = variables.DEPRECIATION_PERIOD
LAMBDA = variables.LAMBDA
event_type_weights = variables.event_points
athlete_countries = variables.athlete_countries
DEPRECIATION_MODEL = variables.DEPRECIATION_MODEL
gender = variables.gender
RANK_DIST = variables.RANK_DIST

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def get_age_weight(race_date_text, ranking_date):
    if DEPRECIATION_MODEL == "linear":
        race_date = dt.strptime(race_date_text, "%m/%d/%Y")
        rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
        days_old = (rank_date.date() - race_date.date()).days
        weight = (DEPRECIATION_PERIOD - days_old) / DEPRECIATION_PERIOD
        return weight
    elif DEPRECIATION_MODEL == "exponential":
        e = 2.71828
        race_date = dt.strptime(race_date_text, "%m/%d/%Y")
        rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
        days_old = (rank_date.date() - race_date.date()).days
        years_old = days_old / 365
        weight = e ** (LAMBDA * years_old)
        return weight


def get_comp_weight(event_type):
    """
    :param event_type: event type as text, ie: "FINA World Cup"
    :return: weight as a float
    """
    weight = float(event_type_weights.weight[event_type_weights.event == event_type])
    return weight


def get_distance_weight(race_dist, units="km"):
    """
    :param race_dist: distance of the race (default is km)
    :param units: 'km' (default) or 'mi'
    :return: weight as a float
    """
    if RANK_DIST == 0:
        weight = 1
    else:
        if units == "mi":
            race_dist = race_dist * 1.60934
        weight = min(race_dist, RANK_DIST) / max(race_dist, RANK_DIST)
    return weight


def label(race_result_file, *args):
    race_data = pd.read_csv(race_result_file)
    race_label = ""
    for arg in args:
        race_label = race_label + str(race_data[arg][0]) + " "
    return race_label.strip()


def update_graph(race_result_file, ranking_date):
    """
    :param ranking_date:
    :param race_result_file: csv file with results from a single race
    :return: adds nodes and edges from race_result_file to existing graph
    """
    # is global G needed? seems to work without it...
    global G

    race_data = pd.read_csv(race_result_file)
    name_list = [name.title() for name in race_data.athlete_name.tolist()]
    age_weight = get_age_weight(race_data.date[0], ranking_date)
    comp_weight = get_comp_weight(race_data.event[0])
    dist_weight = get_distance_weight(race_data.distance[0])
    total_weight = age_weight * comp_weight * dist_weight
    race_label = label(race_result_file, "event", "location", "distance", "date")

    combos = list(combinations(name_list, 2))
    combos = [tuple(reversed(combo)) for combo in combos]

    for combo in combos:
        # from_name = combo[0].title()
        # to_name = combo[1].title()
        # # check for a tie
        # if int(race_data.place[race_data.athlete_name == from_name]) \
        #         == int(race_data.place[race_data.athlete_name == to_name]):
        #     pass
        if combo in G.edges:
            current_weight = G[combo[0]][combo[1]]["weight"]
            new_weight = current_weight + total_weight
            G[combo[0]][combo[1]]["weight"] = new_weight
            G[combo[0]][combo[1]]["race_weights"][race_label] = total_weight

        else:
            label_dict = {
                race_label: total_weight
            }
            G.add_edge(*combo, weight=total_weight, race_weights=label_dict)


def test_predictability(race_result_file):
    """
    :param race_result_file: a new race result csv file to compare against the ranking at that point in time
    :return: adds to correct_predictions (if applicable) and total_matchups running count
    """

    global correct_predictions
    global total_tests

    instance_correct_predictions = 0
    instance_total_tests = 0
    race_label = label(race_result_file, "event", "location", "date", "distance")

    # Optimize test for a subset of the overall ranking.
    from_rank = 4
    to_rank = 5

    ranking_data = pd.read_csv(RANKING_FILE_NAME).iloc[(from_rank - 1):to_rank]
    race_data = pd.read_csv(race_result_file)
    name_list = race_data.athlete_name.tolist()
    combos = list(combinations(name_list, 2))

    for matchup in combos:
        winner_name = matchup[0].title()
        loser_name = matchup[1].title()
        if winner_name in list(ranking_data.name) and loser_name in list(ranking_data.name):
            winner_rank = int(ranking_data["rank"][ranking_data.name == winner_name])
            loser_rank = int(ranking_data["rank"][ranking_data.name == loser_name])
            total_tests += 1
            instance_total_tests += 1
            if winner_rank < loser_rank:
                correct_predictions += 1
                instance_correct_predictions += 1

    try:
        instance_predictability = instance_correct_predictions / instance_total_tests
        # print(f"{race_label} instance_predictability {instance_predictability}")
    except ZeroDivisionError:
        # print(f"cannot calculate predictability for {race_result_file} -- cannot divide by 0")
        pass
    else:
        # instance_predictability = "{:.0%}".format(instance_predictability)
        # print(f"Ranking predictability at {race_label}: {instance_predictability}")


def create_ranking(ranking_date, test=False, comment=False, display_list=0, vis=0, summary=False):

    start = time.time()
    global correct_predictions
    global total_tests
    global G
    global RANKING_FILE_NAME
    G = nx.DiGraph()
    race_count = 0

    # first remove the ranking file that may exist from past function calls
    if os.path.exists(RANKING_FILE_NAME):
        os.remove(RANKING_FILE_NAME)

    # Loop through each race result file. If it's in the date range, update global G with that race's results by
    # calling update_rankings()
    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
        if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
            if comment:
                print(f"Excluding {file}, race is not in date range.")
            else:
                pass
        elif os.path.exists(RANKING_FILE_NAME):
            if test:
                test_predictability(results_file_path)
            if comment:
                print(f"Loading {file}")
            update_graph(results_file_path, ranking_date)
            race_count += 1
            pr_dict = nx.pagerank(G)
            ranking_dict = {
                "name": list(pr_dict.keys()),
                "pagerank": list(pr_dict.values())
            }
            ranking_df = pd.DataFrame(ranking_dict)
            ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
            ranking_df["rank"] = range(1, len(pr_dict) + 1)
            ranking_df.to_csv(RANKING_FILE_NAME, index=False)
        else:
            if test:
                pass
            if comment:
                print(f"Loading {file}")
            update_graph(results_file_path, ranking_date)
            race_count += 1
            pr_dict = nx.pagerank(G)
            ranking_dict = {
                "name": list(pr_dict.keys()),
                "pagerank": list(pr_dict.values())
            }
            ranking_df = pd.DataFrame(ranking_dict)
            ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
            ranking_df["rank"] = range(1, len(pr_dict) + 1)
            ranking_df.to_csv(RANKING_FILE_NAME, index=False)

    if test:
        predictability = correct_predictions / total_tests
        # predictability = "{:.0%}".format(predictability)
        # print(correct_predictions)
        # print(total_tests)
        # print(f"Predictability: {predictability}")
        print(predictability)
        return predictability

    if display_list > 0:
        ranking_data = pd.read_csv(RANKING_FILE_NAME)
        print(ranking_data[ranking_data["rank"] < display_list + 1])

    if vis > 0:

        pr_dict = nx.pagerank(G)

        ranking_dict = {
            "name": list(pr_dict.keys()),
            # "country": [athlete_countries.country[athlete_countries.proper_name == name] for name in
            #                 list(pr_dict.keys())],
            "pagerank": list(pr_dict.values())
        }

        ranking_df = pd.DataFrame(ranking_dict)
        ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
        ranking_df["rank"] = range(1, len(pr_dict) + 1)

        num_of_athletes = vis
        top_athletes = list(ranking_df.name[ranking_df["rank"] < num_of_athletes + 1])
        G = G.subgraph(top_athletes)

        size_map = []
        thicknesses = []
        for name in G.nodes:
            size_map.append(float(ranking_df.pagerank[ranking_df.name == name] * 3000))
        for edge in G.edges:
            thicknesses.append(G[edge[0]][edge[1]]["weight"] * .4)

        nx.draw_networkx(G, node_size=size_map, font_size=8, font_color="red", width=thicknesses,
                         pos=nx.spring_layout(G))
        plt.show()

    end = time.time()

    if summary:
        print(f"New ranking file created: {RANKING_FILE_NAME}")
        print(f"Time to execute: {round((end - start), 2)}s")
        print(f"Races included in ranking: {race_count}")
        print(f"Gender: {gender}")
        print(f"Distance: {RANK_DIST}km")
        print(f"Depreciation period: {DEPRECIATION_PERIOD / 365} years")
        print(f"Depreciation model: {DEPRECIATION_MODEL}")


def alpha_date(date):
    """
    :param date: MM/DD/YYYY
    :return: YYYY_MM_DD
    """
    date = date.replace("/", "_")
    alphadate = date[6:] + "_" + date[:5]
    return alphadate


def unalpha_date(date):
    """
    :param date: YYYY_MM_DD
    :return: MM/DD/YYYY
    """
    uad = date[5:] + "_" + date[:4]
    uad = uad.replace("_", "/")
    return uad


def archive_ranking(ranking_date):
    global G
    G = nx.DiGraph()

    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
        if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
            pass
        else:
            update_graph(results_file_path, ranking_date)

    pr_dict = nx.pagerank(G)

    ranking_dict = {
        "name": list(pr_dict.keys()),
        "pagerank": list(pr_dict.values())
    }

    ranking_df = pd.DataFrame(ranking_dict)
    ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = range(1, len(pr_dict) + 1)
    file_name = f"{alpha_date(ranking_date)}_{gender}_{RANK_DIST}km.csv"
    ranking_df.to_csv(f"{gender}/rankings_archive/{file_name}", index=False)
    print(f"{gender}/rankings_archive/{file_name} archived")


def archive_rankings_range(start_date, end_date, increment=1):
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    rank_dates = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    files_created = 0
    total_files = len(rank_dates)

    for date in rank_dates:
        archive_ranking(date)
        files_created += 1
        progress = files_created / total_files
        print("{:.0%}".format(progress))


def ranking_progression_from_archive(athlete_name, start_date, end_date, increment=1, save=False):
    """
    :param athlete_name:
    :param start_date:
    :param end_date:
    :return: graph showing athlete's ranking on every day between (inclusive) start_date and end_date
    :param increment:
    """

    # Get a list of dates called date_range within the start and end range
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    athlete_name = athlete_name.title()
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    # Loop through each of the dates in date_range and look up the ranking for that date in the archive. Add the date
    # to one list and add the athlete's rank to a separate list. Count loops to track progress.
    dates = []
    ranks = []
    loop_count = 0

    for date in date_range:
        file_name = f"{alpha_date(date)}_{gender}_{RANK_DIST}km.csv"
        ranking_data = pd.read_csv(f"{gender}/rankings_archive/{file_name}")
        ranked_athletes = list(ranking_data.name)
        if athlete_name in ranked_athletes:
            dates.append(date)
            rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
            ranks.append(rank_on_date)
        progress = loop_count / len(date_range)
        loop_count += 1
        print("{:.0%}".format(progress))

    # Create a list of all dates within the date range that the athlete raced, a list of the athlete's rank on that
    # date, and a list of the race names to be used as labels in the graph
    all_race_dates = list(get_results(athlete_name).date)
    race_dates = [date for date in all_race_dates if start_date <= dt.strptime(date, "%m/%d/%Y") <= end_date]
    print(race_dates)

    all_events = list(get_results(athlete_name).event)
    all_locations = list(get_results(athlete_name).location)
    all_dists = list(get_results(athlete_name).distance)
    all_race_labels = [all_events[i] + " " + all_locations[i] + " (" + str(int(all_dists[i])) + "km)" for i in
                       range(len(all_events))]
    race_labels = []

    # Build the list of dates and labels to be used in the graph
    for date in all_race_dates:
        if start_date <= dt.strptime(date, "%m/%d/%Y") <= end_date:
            race_labels.append(all_race_labels[all_race_dates.index(date)])

    # Build the list of ranks to be used in the graph
    race_date_ranks = []

    print(all_race_dates)
    print(all_race_labels)
    print(race_dates)
    print(race_labels)

    print(race_dates)

    for rd in race_dates:
        file_name = f"{alpha_date(rd)}_{gender}_{RANK_DIST}km.csv"
        ranking_data = pd.read_csv(f"{gender}/rankings_archive/{file_name}")
        rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
        race_date_ranks.append(rank_on_date)
        progress = len(race_date_ranks) / len(race_dates)
        print("{:.0%}".format(progress))

    print("Progression dates and ranks:")
    print(dates)
    print(ranks)
    print("Race dates, ranks, and race labels:")
    print(race_dates)
    print(race_date_ranks)
    print(race_labels)

    if save:
        dict = {
            "dates": dates,
            "ranks": ranks
        }
        df = pd.DataFrame(dict)
        df.to_csv(f"{gender}/progressions/{athlete_name} progression.csv")

    # Plot progression dates and ranks in a step chart, plot races on top of that as singular scatter points.
    dates = [dt.strptime(date, "%m/%d/%Y") for date in dates]
    race_dates = [dt.strptime(date, "%m/%d/%Y") for date in race_dates]

    plt.step(dates, ranks, where="post")
    plt.plot(race_dates, race_date_ranks, "o")
    for i, label in enumerate(race_labels):
        plt.text(race_dates[i], race_date_ranks[i], label, rotation=25, fontsize="xx-small")
    plt.ylim(ymin=0.5)
    plt.gca().invert_yaxis()
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("World Ranking")
    start_date = dt.strftime(start_date, "%m/%d/%Y")
    end_date = dt.strftime(end_date, "%m/%d/%Y")
    title_start_date = dt.strftime(dates[0], "%m/%d/%Y")
    title_end_date = dt.strftime(dates[-1], "%m/%d/%Y")
    plt.title(f"{RANK_DIST}km World Ranking Progression: {athlete_name}\n{title_start_date} to {title_end_date}\n")
    plt.show()


def show_results(athlete_name, as_of=dt.strftime(date.today(), "%m/%d/%Y")):
    rows = []

    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        names_list = list(race_data.athlete_name)
        names_list = [name.title() for name in names_list]
        race_data.athlete_name = names_list
        if athlete_name.title() in names_list:
            # calculate the weight of this race and add it to the row:
            age_weight = max(0, get_age_weight(race_data.date[0], as_of))
            comp_weight = get_comp_weight(race_data.event[0])
            dist_weight = get_distance_weight(race_data.distance[0])
            total_weight = age_weight * comp_weight * dist_weight
            row = race_data[race_data.athlete_name == athlete_name.title()]
            row["weight"] = total_weight
            # calculate the WR of the top 10 finishers, average it, and add it to the row:
            top_ten = names_list[0:min(10, len(names_list))]
            rank_file = f"{gender}/rankings_archive/{alpha_date(as_of)}_{gender}_{RANK_DIST}km.csv"
            rank_df = pd.read_csv(rank_file)
            # top_ten_ranks = [int(rank_df["rank"][rank_df["name"] == name]) for name in top_ten
            #                  if name in list(rank_file["name"])]
            # print(int(rank_df["rank"][rank_df["name"] == top_ten[0]]))
            # print(int(rank_df["rank"][rank_df["name"] == top_ten[1]]))
            # print(int(rank_df["rank"][rank_df["name"] == top_ten[2]]))
            top_ten_ranks = []
            for name in top_ten:
                try:
                    rank = int(rank_df["rank"][rank_df["name"] == name])
                except TypeError:
                    pass
                else:
                    top_ten_ranks.append(rank)
            top_ten_avg = sum(top_ten_ranks) / len(top_ten_ranks)
            ranks_list_as_str = ", ".join([str(num) for num in top_ten_ranks])
            row["top 10 avg rank"] = ranks_list_as_str
            rows.append(row)

    df = pd.concat(rows, ignore_index=True)
    # df.sort_values(by="weight", ascending=False).reset_index(drop=True)
    print(df)
    df.to_csv(f"{gender}/show_results/{athlete_name}.csv")


def get_results(athlete_name):
    rows = []

    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        names_list = list(race_data.athlete_name)
        names_list = [name.title() for name in names_list]
        race_data.athlete_name = names_list
        if athlete_name.title() in names_list:
            row = race_data[race_data.athlete_name == athlete_name.title()]
            rows.append(row)
            # print(file)
            # print(row)

    return pd.concat(rows, ignore_index=True)


def show_edges(graph, athlete1, athlete2):
    """
    :param graph: a previously created ranking graph
    :param athlete1: string
    :param athlete2: string
    :return: shows all edges (wins/losses) between to two athletes in the graph that is passed in
    """

    results_dict = {
        "winner": [],
        "event": [],
        "weight": []
    }

    athlete1 = athlete1.title()
    athlete2 = athlete2.title()

    athlete_one_wins = graph[athlete2][athlete1]["race_weights"]

    for (key, value) in athlete_one_wins.items():
        results_dict["winner"].append(athlete1)
        results_dict["event"].append(key)
        results_dict["weight"].append(value)

    athlete_two_wins = graph[athlete1][athlete2]["race_weights"]

    for (key, value) in athlete_two_wins.items():
        results_dict["winner"].append(athlete2)
        results_dict["event"].append(key)
        results_dict["weight"].append(value)

    df = pd.DataFrame(results_dict)
    print(df)


def print_race_labels():
    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        print(label(results_file_path, "event", "location", "date", "distance"), "km")


def compare_place_wr(results_file_path):
    race_data = pd.read_csv(results_file_path)
    race_date = race_data.date[0]
    rank_date = (dt.strptime(race_date, "%m/%d/%Y") - timedelta(days=1)).strftime("%m/%d/%Y")
    create_ranking(rank_date, comment=False)
    places = list(race_data.place)
    athletes = list(race_data.athlete_name)
    athletes = [athlete.title() for athlete in athletes]
    ranking = pd.read_csv(RANKING_FILE_NAME)

    graph_places = []
    graph_athletes = []
    graph_ranks = []

    for i in range(len(places)):
        try:
            rank = int(ranking["rank"][ranking["name"] == athletes[i]])
        except:
            pass
        else:
            graph_places.append(places[i])
            graph_athletes.append(athletes[i])
            graph_ranks.append(rank)

    place_wr_dict = {
        "name": graph_athletes,
        "place": graph_places,
        "rank": graph_ranks
    }

    df = pd.DataFrame(place_wr_dict)
    print(df)

    print(place_wr_dict)

    plt.plot(graph_ranks, graph_places, "o")
    plt.xlabel("World Ranking")
    plt.ylabel("Place")
    title = label(results_file_path, "event", "location", "date", "distance") + "km"
    plt.title(f"{title}")
    plt.show()


def sum_of_edges(graph, athlete):
    weight_dict = {
    }

    for node in graph.nodes:
        # print(node)
        try:
            dict = graph[node][athlete]["race_weights"]
        except KeyError:
            pass
        else:
            # print(dict)
            for key, value in dict.items():
                if key in weight_dict.keys():
                    # pass
                    weight_dict[key] += value
                else:
                    weight_dict[key] = value

    new_dict = {
        "race": list(weight_dict.keys()),
        "sum_of_weights": list(weight_dict.values()),
    }

    df = pd.DataFrame(new_dict)
    print(df.sort_values(by="sum_of_weights", ascending=False).reset_index(drop=True))
    print(f"Sum of edge weights directed at {athlete}: {sum(weight_dict.values())}")


def horse_race_rank(start_date, end_date, num_athletes, increment, type="rank"):
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    athlete_list = []

    for date in date_range:
        file_path = f"{gender}/rankings_archive/{alpha_date(date)}_{gender}_{RANK_DIST}km.csv"
        df = pd.read_csv(file_path)
        all_athletes = list(df.name)
        selected_athletes = all_athletes[0:num_athletes + 1]
        for athlete in selected_athletes:
            if athlete not in athlete_list:
                athlete_list.append(athlete)

    horse_race_dict = {
        "Name": athlete_list
    }

    for date in date_range:
        chart_values = []
        file_path = f"{gender}/rankings_archive/{alpha_date(date)}_{gender}_{RANK_DIST}km.csv"
        df = pd.read_csv(file_path)
        print(file_path)
        for athlete in athlete_list:
            try:
                if type == "ranking":
                    chart_value = int(df["rank"][df["name"] == athlete])
                elif type == "rating":
                    chart_value = float(df["pagerank"][df["name"] == athlete])
            except TypeError:
                if type == "ranking":
                    chart_value = 1000
                elif type == "rating":
                    chart_value = 0
            chart_values.append(chart_value)
        horse_race_dict[date] = chart_values

    df = pd.DataFrame(horse_race_dict)
    df.to_csv("horserace.csv")


def time_diffs(dist, athlete, comp_to_athlete):
    diffs = []

    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        race_dist = race_data.distance[0]
        if athlete in list(race_data.athlete_name) and comp_to_athlete in list(race_data.athlete_name):
            if race_dist == dist or dist == "all":
                main_time = float(race_data["time"][race_data["athlete_name"] == athlete])
                comp_to_time = float(race_data["time"][race_data["athlete_name"] == comp_to_athlete])
                diff = round(main_time - comp_to_time, 2)
                if not (math.isnan(diff)):
                    diffs.append(diff)

    return diffs


def plot_time_diffs(dist, max_diff, athlete_name, *comp_athletes):
    """
    :param dist: number or "all"
    :param max_diff: in seconds, max/min shown on chart
    :param athlete_name: athlete you are comparing to all others in comp_athletes
    :param comp_athletes: athletes that athlete_name is being compared to
    :return: chart
    """

    all_names = []
    all_diffs = []
    all_hues = []
    sb.set_style("darkgrid")

    for comp_athlete in comp_athletes:
        diffs = time_diffs(dist, athlete_name, comp_athlete)
        if len(diffs) > 0:
            for diff in diffs:
                all_names.append(comp_athlete)
                all_diffs.append(diff)
                if diff > 0:
                    win_lose = "lose"
                else:
                    win_lose = "win"
                all_hues.append(win_lose)

    diff_dict = {
        "Competitor": all_names,
        f"Time Difference: {athlete_name} compared to competitors": all_diffs,
        f"Outcome for {athlete_name}": all_hues
    }

    df = pd.DataFrame(diff_dict)
    # print(df)
    chart = sb.stripplot(y="Competitor", x=f"Time Difference: {athlete_name} compared to competitors",
                         hue=f"Outcome for {athlete_name}", linewidth=1, size=7, data=df)
    if dist == "all":
        dist_subtitle = "all race distances"
    else:
        dist_subtitle = f"{dist}km races"
    chart.set(title=f"{athlete_name}'s time differential to various competitors\n{dist_subtitle}, +/- {max_diff}s")
    chart.set_xlim(-max_diff, max_diff)
    chart.invert_xaxis()
    plt.show()


def compare_wr_num_races(ranking_date, comment=False, summary=False):

    start = time.time()
    global correct_predictions
    global total_tests
    global G
    global RANKING_FILE_NAME
    G = nx.DiGraph()
    race_count = 0
    athlete_list = []

    # first remove the ranking file that may exist from past function calls
    if os.path.exists(RANKING_FILE_NAME):
        os.remove(RANKING_FILE_NAME)

    # Loop through each race result file. If it's in the date range, update global G with that race's results by
    # calling update_rankings()
    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
        if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
            if comment:
                print(f"Excluding {file}, race is not in date range.")
            else:
                pass
        elif os.path.exists(RANKING_FILE_NAME):
            if comment:
                print(f"Loading {file}")
            update_graph(results_file_path, ranking_date)
            athlete_list.extend(list(race_data.athlete_name))
            race_count += 1
            pr_dict = nx.pagerank(G)
            ranking_dict = {
                "name": list(pr_dict.keys()),
                "pagerank": list(pr_dict.values())
            }
            ranking_df = pd.DataFrame(ranking_dict)
            ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
            ranking_df["rank"] = range(1, len(pr_dict) + 1)
            ranking_df["qty_races"] = [athlete_list.count(name) for name in list(ranking_df["name"])]
            ranking_df.to_csv(RANKING_FILE_NAME, index=False)
        else:
            if comment:
                print(f"Loading {file}")
            update_graph(results_file_path, ranking_date)
            athlete_list.extend(list(race_data.athlete_name))
            race_count += 1
            pr_dict = nx.pagerank(G)
            ranking_dict = {
                "name": list(pr_dict.keys()),
                "pagerank": list(pr_dict.values())
            }
            ranking_df = pd.DataFrame(ranking_dict)
            ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
            ranking_df["rank"] = range(1, len(pr_dict) + 1)
            ranking_df["qty_races"] = [athlete_list.count(name) for name in list(ranking_df["name"])]
            ranking_df.to_csv(RANKING_FILE_NAME, index=False)

    df = pd.read_csv(RANKING_FILE_NAME)
    ranks = list(df["rank"])
    qty_races = list(df["qty_races"])
    plt.plot(qty_races, ranks, "o")
    plt.gca().invert_yaxis()
    # plt.xlim(xmin=9.5)
    plt.xlabel("Quantity of races in ranking")
    plt.ylabel("World Ranking")
    plt.title(f"Does Racing More Improve Your World Ranking?\n(ranking as of {ranking_date}, "
              f"includes {DEPRECIATION_PERIOD / 365} years of results)")
    plt.show()


    end = time.time()

    if summary:
        print(f"New ranking file created: {RANKING_FILE_NAME}")
        print(f"Time to execute: {round((end - start), 2)}s")
        print(f"Races included in ranking: {race_count}")
        print(f"Gender: {gender}")
        print(f"Distance: {RANK_DIST}km")
        print(f"Depreciation period: {DEPRECIATION_PERIOD / 365} years")
        print(f"Depreciation model: {DEPRECIATION_MODEL}")


def optimization_test(year_start_value, year_end_value, increment):

    dates_to_test = ["01/31/2020", "02/29/2020", "03/31/2020", "04/30/2020", "05/31/2020", "06/30/2020",
                     "07/31/2020", "08/31/2020", "09/30/2020", "10/31/2020", "11/30/2020", "12/31/2020"]

    year_values = [year_start_value]
    keep_going = True
    while keep_going:
        new_num = year_values[-1] + increment
        if new_num <= year_end_value:
            year_values.append(new_num)
        else:
            year_values.append(year_end_value)
            keep_going = False

    dates = []
    opt_year_values = []
    opt_predict_values = []

    for date in dates_to_test:
        year_value_list = []
        predict_value_list = []
        for year_value in year_values:
            # global total_tests
            # global correct_predictions
            DEPRECIATION_PERIOD = 365 * year_value  # reset the depreciation period with new value to test
            year_value_list.append(year_value)
            print(f"date: {date}, years: {year_value}")
            predict_value_list.append(create_ranking(date, test=True))
            # reset the counters after every ranking created
            total_tests = 0
            correct_predictions = 0
        dates.append(date)
        opt_predict_value = max(predict_value_list)
        opt_year_value = year_value_list[predict_value_list.index(opt_predict_value)]
        opt_year_values.append(opt_year_value)
        opt_predict_values.append(opt_predict_value)

    opt_dict = {
        "date": dates,
        "years": opt_year_values,
        "predictability": opt_predict_values
    }

    df = pd.DataFrame(opt_dict)
    print(df)
    df.to_csv(
        f"{gender}/depreciation optimization {alpha_date(dates_to_test[0])} to {alpha_date(dates_to_test[-1])}.csv")


G = nx.DiGraph()
total_tests = 0
correct_predictions = 0


# show_results("Kristof Rasovszky", sortby="weight")
# show_results("Florian Wellbrock")
create_ranking("03/31/2022", test=True, comment=True, summary=True)
# print(G["Kristof Rasovszky"]["Florian Wellbrock"]["race_weights"])
# show_edges(G, "Kristof Rasovszky", "Florian Wellbrock")
# sum_of_edges(G, "Kristof Rasovszky")
# sum_of_edges(G, "Florian Wellbrock")
# archive_rankings_range("04/14/2022", "04/20/2022")


