from pymongo import MongoClient
from config.mongodb_config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

class MongoDBClient:
    def __init__(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[MONGO_DB]
            self.collection = self.db[MONGO_COLLECTION]
            print("✅ MongoDB connected successfully")
        except Exception as e:
            print("❌ MongoDB Connection Error:", e)

    def insert_record(self, data: dict):
        try:
            result = self.collection.insert_one(data)
            return result.inserted_id
        except Exception as e:
            print("❌ Insert Error:", e)
            return None
