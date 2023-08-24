from datetime import datetime

from start_app import db, app


class TimestampMixin:
    created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated = db.Column(db.DateTime, onupdate=datetime.utcnow)


association = db.Table("association",
                       db.Column("day_id", db.Integer, db.ForeignKey("day.id")),
                       db.Column("food_id", db.Integer, db.ForeignKey("food.id"))
                       )


class Food(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20),unique=True)
    protein = db.Column(db.Integer)
    fat = db.Column(db.Integer)
    carbon_hydrate = db.Column(db.Integer)
    calories = db.Column(db.Integer)

    def to_dict(self):
        return  {
            "name" : self.name,
            "far" : self.fat,
            "protein" : self.protein,
            "carbon-hydrate" : self.carbon_hydrate,
            "calories" : self.calories
        }


class Day(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    foods = db.relationship("Food", secondary=association, backref="days")

    def to_dict(self):
        return {
            "id" : self.id,
            "created" : self.created,
            "foods" : [item.to_dict() for item in self.foods]
        }


with app.app_context():
    db.create_all()
