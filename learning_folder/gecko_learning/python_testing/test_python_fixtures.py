import pytest


@pytest.fixture
def orders():
    return []


@pytest.fixture
def add_string(orders):
    orders.append('Hello')


@pytest.fixture
def add_intiger(orders):
    orders.append(2)


def test_order_string(orders, add_string):
    orders.append('world')

    assert ['Hello', 'world'] == orders


def test_order_intiger(orders, add_intiger):
    assert orders == [2]


# ----------- Autouse Fixtures ------------------------#

@pytest.fixture
def order():
    return []


@pytest.fixture
def append_order(order):
    order.append("first")


@pytest.fixture(autouse=True)
def append_auto_fixture(order, append_order):
    order.append('second')


def test_auto_fixture(order):
    assert ['first', 'second'] == order


@pytest.fixture(scope='module')
def scope_module():
    return []


#Da nema return scope module ne bi nista vratilo bilo bi none i ne bi mogao da poredis
@pytest.fixture(scope='module')
def scope_module1(scope_module):
    scope_module.append(1)
    return scope_module


def test_module_1(scope_module):
    assert scope_module == []


def test_module_2(scope_module1):
    assert scope_module1 == [1]


def test_module_3(scope_module):
    scope_module.append(2)
    assert [1, 2] == scope_module


def test_module_4(scope_module):
    scope_module.append(3)
    assert [1, 2, 3] == scope_module


# function: the default scope, the fixture is destroyed at the end of the test.
#
# class: the fixture is destroyed during teardown of the last test in the class.
#
# module: the fixture is destroyed during teardown of the last test in the module.
#
# package: the fixture is destroyed during teardown of the last test in the package.
#
# session: the fixture is destroyed at the end of the test session.


class MailAdmin:

    def create_user(self, index):
        print(f'User is created {index}')
        return MailUser()

    def delete_user(self, index):
        print(f'User is delete {index}')


class MailUser:

    def __init__(self):
        self.messages = []

    def send_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages.clear()


@pytest.fixture
def ret_mail_admin():
    return MailAdmin()


@pytest.fixture
def sending_user(ret_mail_admin):
    user = ret_mail_admin.create_user(1)
    yield user
    print('TearDOWN !!!')
    ret_mail_admin.delete_user(1)


def test_sneding_user(sending_user):

    sending_user.send_message('hello')
    assert sending_user.messages == ['hello']


@pytest.fixture
def ret_order():
    return []


@pytest.fixture(scope='function', params=["hello"])
def adding_nums(request):
    print(request.param)
    return request.param

def test_adding_nums(adding_nums):
    assert adding_nums == 'hello'