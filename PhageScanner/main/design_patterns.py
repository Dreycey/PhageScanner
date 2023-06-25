"""Module for unique design patterns in python."""


class SingletonMeta(type):
    """Metaclass for creating Singleton instances."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Create a Singleton instance.

        Description:
            Ensures the class has not been instantiated.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
