import pandas

import variables

gender = variables.GENDER

consolidated_file_path = f"{gender}/consolidated results/Complete Results {gender}.csv"
race_dict = {
    "athlete_name": [],
    "country": [],
    "date": [],
    "event": [],
    "location": [],
    "distance": [],
    "wetsuit": [],
    "condition": [],
    "field_size": [],
    "time": [],
    "place": [],
}

consolidated_data = pandas.read_csv(consolidated_file_path)
for (index, row) in consolidated_data.iterrows():
    if row.athlete_name == "athlete_name":
        pass

    elif row.athlete_name == "end":
        file_name = row.event
        df = pandas.DataFrame(race_dict)
        df.to_csv(f"{gender}/results/{file_name}.csv", index=False)
        print(f"{file_name}.csv created")
        race_dict["athlete_name"] = []
        race_dict["country"] = []
        race_dict["date"] = []
        race_dict["event"] = []
        race_dict["location"] = []
        race_dict["distance"] = []
        race_dict["wetsuit"] = []
        race_dict["condition"] = []
        race_dict["field_size"] = []
        race_dict["time"] = []
        race_dict["place"] = []

    else:
        race_dict["athlete_name"].append(row.athlete_name)
        race_dict["country"].append(row.country)
        race_dict["date"].append(row.date)
        race_dict["event"].append(row.event)
        race_dict["location"].append(row.location)
        race_dict["distance"].append(row.distance)
        race_dict["wetsuit"].append(row.wetsuit)
        race_dict["condition"].append(row.condition)
        race_dict["field_size"].append(row.field_size)
        race_dict["time"].append(row.time)
        race_dict["place"].append(row.place)





