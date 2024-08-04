from abc import ABC, abstractmethod
import pygame
import math
from self_driving.ai import AI
from self_driving.utils import *
from self_driving.utils import GameAssets


class AbstractCar(ABC):
    def __init__(self, max_vel, rot_vel, is_player_car, start_pos=None, acceleration=0.1) -> None:
        super().__init__()
        self.distances = []
        self.is_player_car = is_player_car
        if not is_player_car:
            self.ai = AI()
            self.dots = GameWindow.PATH.value
        self.curr_vel = 0
        self.curr_rot = 0
        self._max_vel = max_vel
        self._rot_vel = rot_vel
        self.asset = self.IMG.value
        self.pos = self.START_POS if not start_pos else start_pos
        self.acceleration = acceleration

    max_vel, rot_vel = property(lambda self: self._max_vel), property(
        lambda self: self._rot_vel)

    def rotate(self, left: bool = False, right: bool = False) -> None:
        if left and right:
            return
        if left:
            self.curr_rot += self.rot_vel
        elif right:
            self.curr_rot -= self.rot_vel

    def accelerate(self, up: bool = False, down: bool = False) -> None:
        if up and down:
            return
        if up:
            self.curr_vel -= self.acceleration
        elif down:
            self.curr_vel += self.acceleration

    def move(self, up, down) -> None:
        self.accelerate(up, down)
        dx = self.curr_vel * math.sin(math.radians(self.curr_rot))
        dy = self.curr_vel * math.cos(math.radians(self.curr_rot))
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def move_player(self) -> None:
        keys = pygame.key.get_pressed()
        self.rotate(keys[pygame.K_LEFT] or keys[pygame.K_a],
                    keys[pygame.K_RIGHT] or keys[pygame.K_d])
        self.move(keys[pygame.K_UP] or keys[pygame.K_w],
                  keys[pygame.K_DOWN] or keys[pygame.K_s])

    def get_collision_with_track(self, angle, draw=False, max_distance=10000, offset=1, color=(255,255,30)):
        for i in range(0, max_distance, offset):
            dx = i * math.sin(math.radians(angle + self.curr_rot))
            dy = i * math.cos(math.radians(angle + self.curr_rot))
            poi = self.collide(GameAssets.BORDER_MASK.value, dx, dy)
            if poi:
                if draw:
                    x, y = self.pos
                    center_x = x + self.asset.get_width() // 2 
                    center_y = y + self.asset.get_height() // 2
                    
                    pygame.draw.line(GameWindow.WINDOW.value, color, (center_x, center_y), (center_x - dx, center_y - dy), 2)
                return i
        return None


    def car_vision(self, draw=False):
        self.distances = [self.get_collision_with_track(angle, draw) for angle in range(0,360, 45)]
        if draw:
            self.draw(GameWindow.WINDOW.value)
            pygame.display.update()

        return self.distances

    def ai_drive(self, draw=False, log=True) -> None:
        self.prev_state = self.distances + [self.curr_vel, self.curr_rot]
        
        next_moves = self.ai.act(self.car_vision(draw) + [self.curr_vel, self.curr_rot], log)
        if next_moves[0] > 0.66:
            mov = True, False
        elif next_moves[0] < -0.66:
            mov = False, True
        else:
            mov = False, False
        
        if next_moves[1] > 0.66:
            rot = True, False
        elif next_moves[1] < -0.66:
            rot = False, True
        else:
            rot = False, False
        self.move(*mov)
        self.rotate(*rot)
        self.curr_state = self.distances + [self.curr_vel, self.curr_rot]

    def drive(self, draw=False) -> None:
        return self.move_player() if self.is_player_car else self.ai_drive(draw)

    def draw(self, win):
        blit_rotate_center(win, self.asset, self.pos, self.curr_rot)
        
    def compute_reward(self) -> float:
        reward = 0
        
        if self.collide(GameAssets.BORDER_MASK.value) != None:
            reward -= 100
            return reward
        
        next_dot = self.dots[0]
        x, y = self.pos
        x_dot, y_dot = next_dot
        distance = math.sqrt((x - x_dot) ** 2 + (y - y_dot) ** 2)
        if distance <= 10:
            self.dots.pop(0)
            self.dots.append(next_dot)
            reward += 10
        else:
            reward -= distance / 10
        reward += self.curr_vel
        return reward
        
    def register_action(self) -> None:
        if not self.is_player_car:
            reward = self.compute_reward()
            self.ai.apply_reward(reward, self.prev_state, self.curr_state)

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.asset)
        offset = (int(self.pos[0] - x), int(self.pos[1] - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.pos = self.START_POS
        self.curr_rot = 0
        self.curr_vel = 0


class RedCar(AbstractCar):

    IMG = GameAssets.RED_CAR
    START_POS = (180, 200)


class GreenCar(AbstractCar):
    img = GameAssets.GREEN_CAR
    START_POS = (180, 200)


class GreyCar(AbstractCar):
    img = GameAssets.GREY_CAR
    START_POS = (180, 200)


class PurpleCar(AbstractCar):
    img = GameAssets.PURPLE_CAR
    START_POS = (180, 200)


class WhiteCar(AbstractCar):
    img = GameAssets.WHITE_CAR
    START_POS = (180, 200)


class CarFactory:
    player_count = 0

    @staticmethod
    def create_car(car_type, is_player_car, max_vel, rot_vel, start_pos=None):
        if is_player_car and CarFactory.player_count == 0:
            CarFactory.player_count += 1
        elif is_player_car and CarFactory.player_count == 1:
            is_player_car = False
        match car_type:
            case "red":
                return RedCar(max_vel, rot_vel, is_player_car, start_pos)
            case "green":
                return GreenCar(max_vel, rot_vel, is_player_car, start_pos)
            case "grey":
                return GreyCar(max_vel, rot_vel, is_player_car, start_pos)
            case "purple":
                return PurpleCar(max_vel, rot_vel, is_player_car, start_pos)
            case "white":
                return WhiteCar(max_vel, rot_vel, is_player_car, start_pos)
            case _:
                raise ValueError("Invalid car type")
