class PersonMetaClass(type):
    def __new__(cls, name, bases, attrs):
        attrs['print'] = lambda self: print(self._name)
        return type.__new__(cls, name, bases, attrs)


class Person(object, metaclass=PersonMetaClass):
    def __init__(self):
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value


if __name__ == '__main__':
    andy = Person()
    andy.name = 'Andy'

    andy.print()
