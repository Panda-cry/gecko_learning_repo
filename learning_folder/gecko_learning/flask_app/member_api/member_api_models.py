import enum
from datetime import datetime

from member_api_app import app, db


class CardLevel(enum.Enum):
    GOLD = "Gold"
    SILVER = "Silver"
    PLATINUM = "Platinum"


class Member(db.Model):
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30))
    level = db.Column(db.Enum(CardLevel))
    email = db.Column(db.String(30), unique=True)
    created = db.Column(db.DateTime, default=datetime.now())

    def to_dict(self):
        return {
            "name": self.name,
            "email": self.email,
            "level": self.level.value
        }


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
