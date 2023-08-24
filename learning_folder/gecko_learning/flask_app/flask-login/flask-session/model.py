from flask_sqlalchemy import SQLAlchemy
from uuid import uuid4

db =  SQLAlchemy()


def get_uuid():
    return uuid4().hex


class UserFlaskModel(db.Model):
    id = db.Column(db.String(32), primary_key=True, default=get_uuid)
    email = db.Column(db.String(30),unique=True)
    username = db.Column(db.String(30), unique=True)
    password = db.Column(db.Text)
    sesion_id = db.Column(db.String(60))