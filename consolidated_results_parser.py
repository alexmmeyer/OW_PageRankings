import pandas

consolidated_file_path = "Consolidated Results.csv"

race_dict = {
    "athlete_name": [],
    "date": [],
    "event": [],
    "location": [],
    "distance": [],
    "current_type": [],
    "temp": [],
    "field_size": [],
    "place": [],
}

consolidated_data = pandas.read_csv(consolidated_file_path)
for (index, row) in consolidated_data.iterrows():
    if row.athlete_name == "athlete_name":
        pass

    elif row.athlete_name == "end":
        file_name = row.event
        df = pandas.DataFrame(race_dict)
        df.to_csv(f"results/{file_name}.csv",index=False)
        print(f"File {file_name} created")
        race_dict["athlete_name"] = []
        race_dict["date"] = []
        race_dict["event"] = []
        race_dict["location"] = []
        race_dict["distance"] = []
        race_dict["current_type"] = []
        race_dict["temp"] = []
        race_dict["field_size"] = []
        race_dict["place"] = []

    else:
        race_dict["athlete_name"].append(row.athlete_name)
        race_dict["date"].append(row.date)
        race_dict["event"].append(row.event)
        race_dict["location"].append(row.location)
        race_dict["distance"].append(row.distance)
        race_dict["current_type"].append(row.current_type)
        race_dict["temp"].append(row.temp)
        race_dict["field_size"].append(row.field_size)
        race_dict["place"].append(row.place)





