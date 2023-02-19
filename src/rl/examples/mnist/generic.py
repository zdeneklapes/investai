# -*- coding: utf-8 -*-
from typing import List, TypeVar

T = TypeVar('T', covariant=False)

class Animal:
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        print("Woof!")

def animal_sounds(animals: List[T]) -> List[T]:
    for animal in animals:
        animal.make_sound()
    return animals

dogs: List[Dog] = [Dog(), Dog()]
animal_sounds(dogs)  # Type error: Expected List[T], got List[Dog]
