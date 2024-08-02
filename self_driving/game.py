import pygame
from self_driving.car import AbstractCar
from collections import OrderedDict
from self_driving.utils import *

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
        for point in GameWindow.PATH.value:
            pygame.draw.circle(self.window, (255, 0, 0), point, 5)
        pygame.display.update()
    
    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
        
    def step(self):
        clock.tick(GameWindow.FPS.value)
        self.draw_all()
        run = self.process_events()
        for car in self.cars.values(): self.car_step(car)
        return run
    
    def car_step(self, car: AbstractCar):
        car.car_vision(False)
        car.drive()
        self.compute_reward()
        car.apply_reward()
        if car.collide(GameAssets.BORDER_MASK.value) != None:
            car.reset()
        car.draw(self.window)
        self.draw_all_assets()
        self.draw_cars()
        pygame.display.update()
    
    def game_loop(self):
        while run := self.step(): pass
        pygame.quit()

    def compute_reward(self) -> float:
        pass
    