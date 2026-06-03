def pixel_clip(pixel: list[float], width: int, height: int) -> list[float]:
    if not (0 <= pixel[0] <= width - 1):
        return [-1, -1]
    elif not (0 <= pixel[1] <= height - 1):
        return [-1, -1]
    else:
        return pixel


def pixel_to_int(pixel: list[float]) -> list[int]:
    # TODO change list -> tuple
    return [round(pixel[0]), round(pixel[1])]


def check_pixel_invalid(pixel: list) -> bool:
    return all(map(lambda pix: pix != -1, pixel))
