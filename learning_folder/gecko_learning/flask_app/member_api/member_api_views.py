from flask import request
from functools import wraps
from member_api_app import app, db
from member_api_models import Member


def authentication(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        if request.authorization.username != "admin" or request.authorization.password != "admin":
            return "User is not authorised"
        return f(*args, **kwargs)

    return decorator


@app.route('/member', methods=["GET", "POST"])
@authentication
def member():
    if request.method == "GET":
        members = Member.query.all()
        response = [item.to_dict() for item in members]
        return response
    else:
        member = Member()
        post_request = request.get_json()
        member.level = post_request['level']
        member.name = post_request['name']
        member.email = post_request['email']

        db.session.add(member)
        db.session.commit()
        return member.to_dict()


@app.route('/member/<int:id>', methods=["GET", "PUT", "PATCH", "DELETE"])
@authentication
def member_yes(id: int):
    print(id)
    member_get: Member = Member.query.get(id)

    if not member_get:
        return "Member not found"
    if request.method == "GET":
        return member_get.to_dict()

    elif request.method == "PUT":
        data = request.get_json()
        member_get.name = data['name']
        member_get.email = data['email']
        member_get.level = data['level']

        db.session.commit()
        return member_get.to_dict()

    elif request.method == "PATCH":
        data = request.get_json()
        if data['name']:
            member_get.name = data['name']
        elif data['email']:
            member_get.email = data['email']
        elif data['level']:
            member_get.level = data['level']
        db.session.commit()
        return member_get.to_dict()

    else:
        db.session.delete(member_get)
        db.session.commit()
        return f"Member with {id} was deleted"
