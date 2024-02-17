from game import Game
from car import CarFactory

car = CarFactory.create_car("red", True, 5, 5)

game = Game()
game.add_car(car)
game.game_loop()