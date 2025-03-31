import pickle


with open(r"C:\Devansh College\New folder\Health\Medicine-Recommendation-System--1\model\rf.pkl", "rb") as f:
    data = pickle.load(f)


print(data)
