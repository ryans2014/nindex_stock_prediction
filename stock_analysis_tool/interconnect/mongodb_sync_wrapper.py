import pymongo
from datetime import datetime
from interconnect.util import previous_close_utc_time


class SyncResultDocument:
    def __init__(self, symbol: str):
        """
        :param symbol: str, symbol name
        """
        self._symbol = symbol.upper()
        self._document = None

    def fetch(self, db_collection: pymongo.collection.Collection):
        self._document = db_collection.find_one({"symbol": self._symbol})
        return self

    def update(self):
        need_to_update = False
        if self._document is None:
            self._document = {"symbol": self._symbol,
                              "last_update": datetime.utcnow(),
                              "model_version": "v19.9.0",
                              "csv": ""}
            need_to_update = True
        elif self._document["last_update"] < previous_close_utc_time():
            self._document["last_update"] = datetime.utcnow()
            self._document["model_version"] = "v19.9.0"
            need_to_update = True

        if need_to_update:
            pass

        return self

    def push(self, db_collection: pymongo.collection.Collection):
        db_collection.update_one({"symbol": self._symbol}, self._document)
        return self


# client = pymongo.MongoClient(
# "mongodb+srv://ryansu2011:susu1021@hispredict-dbzhi.mongodb.net/test?retryWrites=true&w=majority")
# db = client.histresult
# collection = db.results
