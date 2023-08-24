from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:postgrespw@localhost:55000"

db = SQLAlchemy(app)
migrate = Migrate(app, db, compare_type=True)


class Phone(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(40), unique=True)
    serial_number = db.Column(db.String(30), unique=True)


if __name__ == "__main__":
    app.run(debug=True)
