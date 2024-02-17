import pygame
import time
import math
from car import AbstractCar
from collections import OrderedDict
from utils import *

pygame.display.set_caption('Self Driving Car')
clock = pygame.time.Clock()


class Game:

    id_car = 0

    def __init__(self):
        self.window = GameWindow.WINDOW.value
        self.assets = OrderedDict(**{
            'grass': (GameAssets.GRASS, 0, 0),
            'track': (GameAssets.TRACK, 0, 0),
            'finishLine': (GameAssets.FINISH, 130, 200),
            'border': (GameAssets.BORDER, 0, 0),
        })

        self.cars: OrderedDict[int, AbstractCar] = OrderedDict()

    def add_car(self, car):
        self.id_car += 1
        self.cars[self.id_car] = car

    def draw_asset(self, asset: GameAssets, x, y):
        self.window.blit(asset.value, (x, y))

    def draw_all_assets(self):
        for asset in self.assets.values():
            self.draw_asset(*asset)
    

    def draw_cars(self):
        for car in self.cars.values():
            car.draw(self.window)

    def draw_all(self):
        self.draw_all_assets()
        self.draw_cars()
        pygame.display.update()
            
    def game_loop(self):
        run = True
        while run:
            clock.tick(GameWindow.FPS.value)
            
            self.draw_all()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            for car in self.cars.values():
                car.drive()
                car.car_vision(True)
                if car.collide(GameAssets.BORDER_MASK.value) != None:
                    car.reset()
                car.draw(self.window)
                self.draw_all_assets()
                self.draw_cars()
                pygame.display.update()

        pygame.quit()
