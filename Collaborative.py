import sys
import math
import os
class Collaborate_Filter:
    def __init__(self, input_file_name, user_id, movie, k):
        self.input_file_name = input_file_name
        self.user_id = user_id
        self.movie = movie
        self.k = k
        self.dataset = None
        self.uu_dataset = None
        self.ii_dataset = None

    def start(self):
 # check file exist and if it's a file or dir
        if not os.path.isfile(self.input_file_name):
            self.quit("Input file doesn't exist or it's not a file")
            # load data
        self.dataset, self.uu_dataset, self.ii_dataset = self.load_data(self.input_file_name)
        # check if user exist
        users = self.uu_dataset.keys()
        if self.user_id not in users:
            self.quit("User ID doesn't exist")

    def Person_corealtion(self, user1, user2):
        result = 0.0
        user1_data = self.uu_dataset[user1]
        user2_data = self.uu_dataset[user2]
        rx_avg = self.average_rating_user(user1_data)
        ry_avg = self.average_rating_user(user2_data)
        sxy = self.items_common(user1_data, user2_data)
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
    
    def average_rating_user(self, user_data):
        avg_rating = 0.0
        size = len(user_data)
        for (movie, rating) in user_data.items():
            avg_rating += float(rating)
        avg_rating /= size * 1.0
        return avg_rating
    
    def items_common(self, user1_data, user2_data):
        result = []
        ht = {}
        for (movie, rating) in user1_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (movie, rating) in user2_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (k, v) in ht.items():
            if v == 2:
                result.append(k)
        return result
    
    def knn(self, user, k):
        neighbors = []
        result = []
        for (user_id, data) in self.uu_dataset.items():
            if user_id == user:
                continue
            upc = self.Person_corealtion(user, user_id)
            if upc != 1000:
                neighbors.append([user_id, upc])
        sorted_neighbors = sorted(neighbors, key=lambda neighbors: (neighbors[1], neighbors[0]),reverse=True) # - for desc sort
        for i in range(k):
            if i >= len(sorted_neighbors):
                break
            result.append(sorted_neighbors[i])
        return result
    
    def predict(self, user, item, knn):
        valid_neighbors = self.check_validation_nbr(item,knn)
        if not len(valid_neighbors):
            return 0.0
        top_result = 0.0
        bottom_result = 0.0
        for neighbor in valid_neighbors:
            neighbor_id = neighbor[0]
            neighbor_similarity = neighbor[1] # Wi1
            rating = self.uu_dataset[neighbor_id][item] # rating i,item
            top_result += neighbor_similarity * rating
            bottom_result += neighbor_similarity
        result = top_result / bottom_result
        self.prints(valid_neighbors, result)
        return result

    def check_validation_nbr(self, item, knn):
        result = []
        for neighbor in knn:
            neighbor_id = neighbor[0]
            # print item
            if item in self.uu_dataset[neighbor_id].keys():
                result.append(neighbor)
        return result
    def load_data(self, input_file_name):
        input_file = open(input_file_name, 'r')
        dataset = []
        uu_dataset = {}
        # print(uu_dataset)
        ii_dataset = {}
        for line in input_file:
            row = str(line)
            row = row.split(",")
            dataset.append(row[:-1])
            uu_dataset.setdefault(row[0], {})
            # print(uu_dataset,'inside the loop')
            uu_dataset[row[0]].setdefault(row[1], float(row[2]))
            # print(uu_dataset,'inside the loop2')
            ii_dataset.setdefault(row[1], {})
            ii_dataset[row[1]].setdefault(row[0], float(row[2]))
            # print(uu_dataset,'inside the loop2')
            # print(dataset[0:5])
            # print(uu_dataset[0:5])
            # print(ii_dataset[0:5])
        return dataset, uu_dataset, ii_dataset
    def prints(self, knn, prediction):
        for neighbor in knn:
            print("Neighours-->",neighbor[0],"Ratings->" ,neighbor[1])
        print("\n")
        print("Predictions _Ratings:-->",prediction)
        
if __name__ == '__main__':
    # publish
    input_file_name = "exp_7_rating.csv" # input file location with name
    user_id = input('userId') # user id
    movie = input("movie_name") # movie id
    k = int(input("enter k")) # k neighbors
    
    cf = Collaborate_Filter(input_file_name, user_id, movie, k)
    cf.start()
    
    knn = cf.knn(user_id, k)