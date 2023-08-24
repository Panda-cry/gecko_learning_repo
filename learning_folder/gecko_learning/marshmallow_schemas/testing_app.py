from mars_schemas.entityUser import UserSchema, BlogSchema, FirstSchema, SecondSchema, MapperSchema
from datetime import datetime, date

# user_one_date = datetime(2022, 12, 28)
# user_one = {
#     "name": "Petar",
#     "email": "petar.canic@55gmail.com",
#     #"created_at": user_one_date,
#     "signed_in": "True"
# }
# blog_one = {
#     "title": "FIRST BLOG",
#     "author": {
#         "name": "Petar",
#         "email": "petar.canic@55gmail.com",
#         #"created_at": user_one_date,
#         "signed_in": "True"
#     }
# }
# bblog_serialized = BlogSchema().load(blog_one)
# print(bblog_serialized)
# user = UserSchema().dump(user_one)
# print(user)
# user_ser = UserSchema().dump(user_one)
# print(user_ser)
# string_users_serialized = UserSchema().dumps(user_one)
# print(string_users_serialized)

# user_again = UserSchema().load(users_serialized,partial=('email',))
# print(user_again)

# user_from_string = UserSchema().loads(string_users_serialized)
# print(user_from_string)


map = {
    "mapper_field": {"type": "second","data" : {'act':"1233122"}}
}

data = MapperSchema()
print("byydydaysd")
data = data.load(map)
data = MapperSchema().dump(map)

print(data)
