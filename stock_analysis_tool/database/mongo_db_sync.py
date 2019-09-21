import pymongo
from datetime import datetime, timezone, timedelta

client = pymongo.MongoClient("mongodb+srv://ryansu2011:susu1021@hispredict-dbzhi.mongodb.net/test?retryWrites=true&w=majority")
db = client.histresult
collection = db.results


def previous_close_utc_time() -> datetime:
    """
    :return: previosu trading close time (assume 4:30pm close at EAT) (return tiem is in UTC)
    """
    def utc(est: datetime) -> datetime:
        return est + timedelta(hours=4)

    est_now = datetime.utcnow() - timedelta(hours=4)
    today_close = datetime(year=est_now.year,
                           month=est_now.month,
                           day=est_now.day,
                           hour=16,
                           minute=30)
    if est_now > today_close and est_now.weekday() <= 5:
        return utc(today_close)

    previous_close = today_close - timedelta(hours=24)
    while previous_close.weekday() > 5:
        previous_close = previous_close - timedelta(hours=24)
    return utc(previous_close)


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






