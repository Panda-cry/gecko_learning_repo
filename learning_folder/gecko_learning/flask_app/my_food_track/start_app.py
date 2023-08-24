from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:postgrespw@localhost:55000"

db = SQLAlchemy(app)

from views import *


if __name__ == "__main__":
    app.run(debug=True, port=5001)
