from marshmallow import Schema,fields
import uuid

class InformationSchema(Schema):
    id = fields.UUID(default=uuid.uuid4())
    message = fields.String()


class ErrorSchema(Schema):
    id = fields.UUID(required=True, default=uuid.uuid4())
    message = fields.String()



