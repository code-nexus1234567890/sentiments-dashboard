from pymongo import MongoClient

# Connect to MongoDB Atlas
def get_db():
    client = MongoClient("mongodb+srv://<ayushmishra180904>:<H45C8uUOnKzZzyOd>@cluster0.abcd.mongodb.net/sentimentdb")
    db = client["sentimentdb"]
    return db

# Add user
def add_user(username, password):
    db = get_db()
    users = db["users"]
    if users.find_one({"ayushmishra180904": username}):
        return False
    users.insert_one({"ayushmishra180904": username, "H45C8uUOnKzZzyOd": password})
    return True

# Login check
def login_user(username, password):
    db = get_db()
    users = db["users"]
    return users.find_one({"ayushmishra180904": username, "H45C8uUOnKzZzyOd": password})
