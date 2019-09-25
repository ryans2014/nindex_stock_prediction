from motor import motor_asyncio
from datetime import datetime
from serving.util import previous_close_utc_time
import asyncio


db_collection = None


def connect():
    global db_collection
    clnt = motor_asyncio.AsyncIOMotorClient(
        "mongodb+srv://ryansu2011:susu1021@hispredict-dbzhi.mongodb.net/test?retryWrites=true&w=majority")
    dbs = clnt.histresult
    db_collection = dbs.results


class AsyncResultDocument:
    def __init__(self, symbol: str):
        """
        :param symbol: str, symbol name
        """
        self._symbol = symbol.upper()
        self._document = None
        self._need_update = False
        self._does_not_exist = False

    async def fetch(self):
        self._document = await db_collection.find_one({"symbol": self._symbol})

    def update(self):
        self._need_update = False
        if self._document is None:
            self._document = {"symbol": self._symbol,
                              "last_update": datetime.utcnow(),
                              "model_version": "v19.9.0",
                              "csv": ""}
            self._need_update = True
            self._does_not_exist = True
        elif self._document["last_update"] < previous_close_utc_time():
            self._document["last_update"] = datetime.utcnow()
            self._document["model_version"] = "v19.9.0"
            self._need_update = True

    def need_update_csv(self):
        return self._need_update

    def set_csv(self, csv_string: str):
        self._document["csv"] = csv_string

    def get_csv(self) -> str:
        return self._document["csv"]

    async def push(self):
        if self._does_not_exist:
            db_collection.insert_one(self._document)
        else:
            await db_collection.update_one({"symbol": self._symbol},
                                           {"$set": {
                                                   "last_update": datetime.utcnow(),
                                                   "model_version": self._document["model_version"],
                                                   "csv": self.get_csv()}})


if __name__ == "__main__":
    client = motor_asyncio.AsyncIOMotorClient(
        "mongodb+srv://ryansu2011:susu1021@hispredict-dbzhi.mongodb.net/test?retryWrites=true&w=majority")
    db = client.histresult
    collection = db.results
    d1 = AsyncResultDocument("AAPL")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(d1.fetch())
