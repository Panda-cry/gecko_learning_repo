from datetime import timedelta
from functools import wraps

from flask import Flask, request, jsonify
from flask_jwt_extended import get_jwt_identity, create_access_token, jwt_required, JWTManager, current_user, \
    create_refresh_token, verify_jwt_in_request, get_jwt
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['DEBUG'] = True
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:postgrespw@localhost:55000"
app.config['JWT_SECRET_KEY'] = "this_is_my_secret"
jwt = JWTManager(app)
db = SQLAlchemy(app)


class FlaskTokenMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30))

    # # NOTE: In a real application make sure to properly hash and salt passwords
    # from hmac import compare_digest
    # def check_password(self, password):
    #     return compare_digest(password, "password")


with app.app_context():
    db.create_all()


@jwt.additional_claims_loader
def add_claims_to_access_token(user):
    # Ovde moze if pa neke role da se predstave
    # tj neke claimove da damo
    if "c" in user.username:
        return {
            "is_admin" : True
        }
    else:
        return {
            "is_admin" : False
        }


def admin_required():
    def wrapper(fn):
        wraps(fn)

        def decorator(*args, **kwargs):
            verify_jwt_in_request()
            jwt = get_jwt()

            if jwt['is_admin']:
                return fn(*args, **kwargs)
            else:
                return  jsonify("Only for admins"), 403

        return decorator
    return wrapper

# ovo je da se prosledi ceo user kada se kreira token i sta ce se sve encode
@jwt.user_identity_loader
def identity_loader(user):
    print("Second")
    return user.username


# ovo je za dobijanje user kada se decode jwt
@jwt.user_lookup_loader
def user_lookup_callback(_jwt_header, jwt_data):
    print("First")
    username = jwt_data['sub']
    return FlaskTokenMember.query.filter_by(username=username).first()


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.get_json().get('username')
        user: FlaskTokenMember = FlaskTokenMember.query.filter_by(username=username).first()

        if user:
            access_token = create_access_token(identity=user)
            # za fresh dodati parametar fresh=True
            refresh_token = create_refresh_token(identity=user)
            return jsonify(access_token=access_token, refresh_token=refresh_token)
        else:
            return f"Database doesn't have that specific user {username}"


@app.route('/index')
@jwt_required()
def index():
    current_user1 = get_jwt_identity()
    return jsonify(id=current_user.id, username=current_user.username)


@app.route('/refresh')
@jwt_required(refresh=True)
def refresh():
    access_token = create_access_token(current_user)
    # fresh = False
    # moze i ovako da se kreira create_access_token(identity, fresh=datetime.timedelta(minutes=15))
    return jsonify(access_token=access_token)


@app.route('/change')
@jwt_required(fresh=True)
def change():
    current_user1 = get_jwt_identity()
    return jsonify(message="We need fresh token")



@app.route('/admin')
@admin_required()
def admin():
    return jsonify(message="I am administrator")

if __name__ == "__main__":
    app.run()

# In its simplest form, there is not much to using this extension. You use create_access_token() to make JSON Web Tokens,
# jwt_required() to protect routes, and get_jwt_identity() to get the identity of a JWT in a protected route.
# Mozda je bolje slati JWT kao cookie jer je cookie samo na https i moze da se doda http_Read_only
# Pojam refresh tokena je da imamo obican i refresh token ako je jedan token expired sa refresh token ga ponovimo
# ali nor ako se menja lozinak ili tako nesto sto je vazno potrebno je da imammo fresh token


# Using an `after_request` callback, we refresh any token that is within 30
# minutes of expiring. Change the timedeltas to match the needs of your application.
# @app.after_request
# def refresh_expiring_jwts(response):
#     try:
#         exp_timestamp = get_jwt()["exp"]
#         now = datetime.now(timezone.utc)
#         target_timestamp = datetime.timestamp(now + timedelta(minutes=30))
#         if target_timestamp > exp_timestamp:
#             access_token = create_access_token(identity=get_jwt_identity())
#             set_access_cookies(response, access_token)
#         return response
#     except (RuntimeError, KeyError):
#         # Case where there is not a valid JWT. Just return the original response
#         return response
