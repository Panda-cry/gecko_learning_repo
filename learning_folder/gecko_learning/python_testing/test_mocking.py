import mocking


def mock_request():
    return {'data': 'data'}


def convert_to_json(content):
    return content


def test_mocking_functions(monkeypatch):
    monkeypatch.setattr(mocking, 'send_request_to_get_users', mock_request)
    monkeypatch.setattr(mocking, 'convert_to_json_readable', convert_to_json)
    data = mocking.send_request_to_get_users()
    input = mocking.convert_to_json_readable(data)
    assert {'data': 'data'} == data
    assert {'data': 'data'} == input
