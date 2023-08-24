from datetime import datetime

from app import db


class TimestampMixin:
    created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated = db.Column(db.DateTime, onupdate=datetime.utcnow)


class ThatDay(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    foods = db.relationship("Food")


foods = db.Table(
    "foods",
    db.Column('food_id', db.Integer, db.ForeignKey('food.id'), primary_key=True),
    db.Column('day_id', db.Integer, db.ForeignKey('thatday.id'), primary_key=True)
)


class Food( db.Model):
    id = db.Column(db.Integer, primary_key=True)
    protein = db.Column(db.Integer)
    fat = db.Column(db.Integer)
    carbon_hydrate = db.Column(db.Integer)
    calories = db.Column(db.Integer)
