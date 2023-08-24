from functools import wraps
from uuid import uuid4

from flask import Flask, request, jsonify, session
#ovo sesion stavlja nesto u cookie :D
from flask_bcrypt import Bcrypt

from model import db, UserFlaskModel

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:postgrespw@localhost:55000"
app.config['SECRET_KEY'] = "this_is_my_secret"

crypt = Bcrypt(app)
# nakon generisanja password treba decode da se odradi :D

with app.app_context():
    db.init_app(app)
    db.create_all()


def session_required():
    def wrapped(fn):
        wraps(fn)

        def decorator(*args, **kwargs):

            user = UserFlaskModel.query.filter_by(sesion_id=session['session_id']).first()
            if user is None:
                return jsonify(message="Unauthorized"), 401
            return fn(*args, **kwargs)
        return decorator

    return wrapped


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()

        user: UserFlaskModel = UserFlaskModel.query.filter_by(username=data['username']).first()

        if user is None:
            return jsonify(message="Current user doesn't exists"), 401

        if crypt.check_password_hash(user.password, data['password']):
            return jsonify(message="Password is incorrect"), 401

        user.sesion_id = uuid4().hex
        db.session.commit()

        session['session_id'] = user.sesion_id

        return jsonify(message="Hello {}".format(user.username))


@app.route('/info/<string:name>')
@session_required()
def info(name):
    return jsonify(message="Info about user that is created this service",user=name),200


if __name__ == "__main__":
    app.run(port=5000)
