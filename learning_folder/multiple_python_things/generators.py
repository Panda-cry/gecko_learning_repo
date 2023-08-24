

def generators_functions(input_directory):
    with open(input_directory) as input:
        data = input.readline()
        while data:
            yield data
            data = input.readline()



x = generators_functions('input.txt')

# for item in range(5):
#     print(next(x))


def while_generator(input):
    while input < 10:
        yield input
        input += 1

yea = while_generator(4)



def baz():
    for i in range(10):
        yield i

def bar():
    for i in range(5):
        yield i

def foo():
    yield from bar()
    yield from baz()

for v in foo():
    print(v)