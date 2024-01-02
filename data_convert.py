# convert
# data/gamer_full.json
# from
# [
#     {
#         "username": "Cpl_iPatch",
#         "content": "everyone come to 700 50k its my base",
#         "date": "09/02/2020 2:14 AM"
#     },
# ...

# to
# [
#     {
#         "username": "Cpl_iPatch",
#         "prompt": "everyone come to 700 50k its my base",
#         "date": "09/02/2020 2:14 AM"
#     },
# ...

import json

with open("data/gamer_full.json") as f:
    data = json.load(f)

for i in range(len(data)):
    data[i]["prompt"] = data[i]["content"]
    del data[i]["content"]

with open("data/gamer_full_formatted.json", "w") as f:
    json.dump(data, f)

