from flask import request

from start_app import app, db
from start_models import Food, Day


@app.route('/')
def index():
    return "Hello"


@app.route('/add_food', methods=["POST"])
def add_food():
    food = Food()
    data = request.get_json()
    food.fat = data['fat']
    food.protein = data['protein']
    food.calories = data['calories']
    food.carbon_hydrate = data.get('carbon_hydrate')
    food.name = data.get('name')

    db.session.add(food)
    db.session.commit()

    return food.to_dict()


@app.route('/add_day', methods=["POST"])
def add_day():
    day = Day()
    db.session.add(day)
    db.session.commit()
    return day.to_dict()


@app.route('/connect_food_to_day/<int:food_id>/<int:day_id>')
def connect_food_to_day(food_id: int, day_id: int):
    food = Food.query.get(food_id)
    day = Day.query.get(day_id)

    if food and day:
        day.foods.append(food)
        db.session.commit()
        return day.to_dict()
    else:
        return "Something went wrong"


@app.route('/calories_sum/<int:day_id>')
def calories_sum(day_id: int):
    day = Day.query.get(day_id)

    if day:
        count_calories = sum([item.calories for item in day.foods])
        return f"Sum calories is : {count_calories}"
    else:
        return "We dont have that day"


@app.route('/count_days/<int:food_id>')
def count_days(food_id):
    food = Food.query.get(food_id)
    if food:
        days_count = len(food.days)
        return f"Count of days is {days_count}"
    else:
        return "Something is messy"
