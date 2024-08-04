import pygame
from self_driving.car import AbstractCar
from collections import OrderedDict
from self_driving.utils import *

pygame.display.set_caption('Racing Game')
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

    def add_car(self, car) -> None:
        self.id_car += 1
        self.cars[self.id_car] = car

    def draw_asset(self, asset: GameAssets, x, y) -> None:
        self.window.blit(asset.value, (x, y))

    def draw_all_assets(self) -> None:
        for asset in self.assets.values():
            self.draw_asset(*asset)
    

    def draw_cars(self) -> None:
        for car in self.cars.values():
            car.draw(self.window)

    def draw_all(self) -> None:
        self.draw_all_assets()
        for point in GameWindow.PATH.value:
            pygame.draw.circle(self.window, (255, 0, 0), point, 5)
        self.draw_cars()
        pygame.display.update()
    
    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
        
    def step(self) -> bool:
        clock.tick(GameWindow.FPS.value)
        self.draw_all()
        run = self.process_events()
        for car in self.cars.values(): self.car_step(car)
        return run
    
    def car_step(self, car: AbstractCar) -> None:
        car.car_vision(False)
        car.drive()
        car.compute_and_apply_reward()
        if car.collide(GameAssets.BORDER_MASK.value) != None:
            car.reset()
        car.draw(self.window)
        self.draw_all()
    
    def game_loop(self):
        while run := self.step(): pass
        pygame.quit()
