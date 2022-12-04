
import math

k = 7


class ColloborativeFilter:

    def __init__(self, file_name, k):
        self.file_name = file_name
        self.k = k
        self.user_dataset, self.item_dataset = self.loaddata()

    def loaddata(self):
        file = open(self.file_name, "r")
        lines = file.readlines()[1:]
        user_dataset = {}
        item_dataset = {}

        for line in lines:
            # line = lines[0]
            row = str(line).split(",")
            userid = int(row[0])
            movieid = int(row[1])
            rating = float(row[2])

            user_dataset.setdefault(userid, {})
            user_dataset[userid].setdefault(movieid, rating)

            item_dataset.setdefault(int(row[1]), {})
            item_dataset[movieid].setdefault(userid, rating)
        return user_dataset, item_dataset

    def user_average_rating(self, userid):
        total = 0.0
        allratings = self.user_dataset[userid].items()
        for (_, rating) in allratings:
            total += rating
        return total/(len(allratings)*1.0)

    def common_items(self, userid1, userid2):
        items = {}
        for (movie, _) in self.user_dataset[userid1].items():
            items.setdefault(movie, 0)
            items[movie] += 1
        for (movie, _) in self.user_dataset[userid2].items():
            items.setdefault(movie, 0)
            items[movie] += 1

        result = []
        for (k, v) in items.items():
            if (v == 2):
                result.append(k)
        return result

    def pearson_correlation(self, active_userid, other_userid):
        result = 0.0
        user1_data = self.user_dataset[active_userid]
        user2_data = self.user_dataset[other_userid]
        rx_avg = self.user_average_rating(active_userid)
        ry_avg = self.user_average_rating(other_userid)
        sxy = self.common_items(active_userid, other_userid)
        top_result = 0.0
        bottom_left_result = 0.0
        bottom_right_result = 0.0
        for item in sxy:
            rxs = user1_data[item]
            rys = user2_data[item]
            top_result += (rxs - rx_avg) * (rys - ry_avg)
            bottom_left_result += pow((rxs - rx_avg), 2)
            bottom_right_result += pow((rys - ry_avg), 2)
        bottom_left_result = math.sqrt(bottom_left_result)
        bottom_right_result = math.sqrt(bottom_right_result)
        if bottom_right_result == 0:
            return 1000
        result = top_result / (bottom_left_result * bottom_right_result)
        return result

    def k_nearest_neighbours(self, active_userid):
        neighbours = []
        for (userid, data) in self.user_dataset.items():
            if (active_userid == userid):
                continue
            upc = self.pearson_correlation(active_userid, userid)
            if upc != 1000:
                neighbours.append([userid, upc])

        sorted_neighbours = sorted(
            neighbours, key=lambda neighbour: neighbour[1], reverse=True)

        # select k neighbours
        return neighbours[:min(k, len(sorted_neighbours))]

    def validate_neighbours(self, itemid, neighbours):
        # check if given neighbours has given rating on itemid or not
        result = []
        for neighbour in neighbours:
            neighbourid = neighbour[0]
            if itemid in self.user_dataset[neighbourid].keys():
                result.append(neighbour)
        return result

    def predict(self, userid, itemid):
        if userid not in self.user_dataset.keys():
            print("Userid not found!!")
            return
        if itemid not in self.item_dataset.keys():
            print("Movieid not found!!")
            return
        neighbours = self.k_nearest_neighbours(userid)
        neighbours = self.validate_neighbours(itemid, neighbours)
        top_result = 0.0
        bottom_result = 0.0
        for neighbor in neighbours:
            neighbor_id = neighbor[0]
            neighbor_similarity = neighbor[1]  # Wi1
            rating = self.user_dataset[neighbor_id][itemid]  # rating i,item
            top_result += neighbor_similarity * rating
            bottom_result += neighbor_similarity
        result = top_result / bottom_result
        return result


def main():
    filename = "colloborative-filtering.csv"
    cf = ColloborativeFilter(filename, k)
    # userid = input("Userid: ")
    # itemid = input("MovieId: ")
    userid = 4
    itemid = 48516
    prediction = cf.predict(userid, itemid)
    print("Prediction-> ", prediction)


main()
