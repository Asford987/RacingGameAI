import pygame
import enum


def resize(asset, factor_width, factor_height=None):
    if factor_height is None:
        factor_height = factor_width
    new_size = round(asset.get_width() *
                     factor_width), round(asset.get_height() * factor_height)
    return pygame.transform.scale(asset, new_size)


def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(
        center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)


class GameAssets(enum.Enum):

    RED_CAR = resize(pygame.image.load(
        'assets/imgs/red-car.png'), 0.55)

    GREEN_CAR = resize(pygame.image.load(
        'assets/imgs/green-car.png'), 0.55)

    GREY_CAR = resize(pygame.image.load(
        'assets/imgs/grey-car.png'), 0.55)

    PURPLE_CAR = resize(pygame.image.load(
        'assets/imgs/purple-car.png'), 0.55)

    WHITE_CAR = resize(pygame.image.load(
        'assets/imgs/white-car.png'), 0.55)

    GRASS = resize(pygame.image.load(
        'assets/imgs/grass.jpg'), 2.5)

    TRACK = resize(pygame.image.load(
        'assets/imgs/track.png'), 0.9)

    BORDER = resize(pygame.image.load(
        'assets/imgs/track-border.png'), 0.9)

    FINISH = pygame.image.load('assets/imgs/finish.png')

    BORDER_MASK = pygame.mask.from_surface(BORDER)


class GameWindow(enum.Enum):
    WIDTH, HEIGHT = GameAssets.TRACK.value.get_size()
    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
    PATH = [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
            (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]
    FPS = 60
