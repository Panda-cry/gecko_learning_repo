from flask import Flask, request
from flask_login import UserMixin, LoginManager, login_required, login_user, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:postgrespw@localhost:55000"
app.config['SECRET_KEY'] = "this_is_my_secret"
#app.config['REMEMBER_COOKIE_DURATION'] = 20 - sec
login_manager = LoginManager(app)
db = SQLAlchemy(app)


class FlaskMember(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30))
    session_id = db.Column(db.String(50))


@login_manager.user_loader
def load_user(user_id):
    return FlaskMember.query.get(int(user_id))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        data = request.get_json()
        user = FlaskMember.query.filter_by(username=data.get('username')).first()
        login_user(user)
        #pred user ide remember = True
    return "TADA"


@app.route("/home")
@login_required
def home():
    return "We are protected {}".format(current_user.username)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return "We are logout"


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run()
