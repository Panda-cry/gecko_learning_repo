import typing

import marshmallow.validate
from marshmallow import Schema, fields, pre_load, post_load, validates


def validate_name(name):
    if len(name) > 6:
        raise marshmallow.validate.ValidationError("Name cannot be above 6 characters")


class UserSchema(Schema):
    name = fields.Str(validate=validate_name)
    email = fields.Email(required=True, load_only=True)
    # created_at = fields.DateTime()
    signed_in = fields.Boolean()

    @validates("signed_in")
    def validate_signed_in(self, value):
        if value:
            print(f'Validate signed_in is TRUE !!!')
        else:
            print('Validate signed_in is FALSE')

    def _validate_name(self, name):
        print("buahahahahah")

    @pre_load
    def testing_load_functions(self, data, **kwargs):
        # data.pop('created_at')
        print("PRE_LOAD USER SCHEMA HAPPENED HERE !!!")
        return data

    @post_load
    def testing_post_load_functions(self, data, **kwargs):
        print("POST LOAD USER SCHEMA HAPPENED HERE !!!")
        return data


class BlogSchema(Schema):
    title = fields.String()
    author = fields.Nested("UserSchema")


class FirstSchema(Schema):
    title = fields.String()


class SecondSchema(Schema):
    act = fields.String()


class Mapper(fields.Field):

    def __init__(self, key, maps, fallback_schema, **kwargs):
        self.key = key
        self.maps = maps
        self.fallback_schema = fallback_schema
        super().__init__(**kwargs)

    def _serialize(
            self, value, attr, obj, **kwargs
    ):
        breakpoint()
        schema = self._get_schema(value[self.key])
        return schema().dump(value['data'])

    def _deserialize(
            self,
            value,
            attr,
            data,
            **kwargs,
    ):
        schema = self._get_schema(value[self.key])
        return schema().load(value['data'])

    def _get_schema(self, key: str) -> Schema:
        if key in self.maps:
            return self.maps[key]
        else:
            print("We dont have that key")


class MapperSchema(Schema):
    _map = {'first': FirstSchema, "second": SecondSchema}
    mapper_field = Mapper("type", maps=_map, fallback_schema=SecondSchema, required=True)
