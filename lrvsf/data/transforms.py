from torchvision.transforms import transforms as tf
import torchvision.transforms.functional as F


class SquarePad:
    def __init__(self, color):
        self.col = color

    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, self.col, "constant")


def train_tf(size):
    return tf.Compose(
        [
            SquarePad(255),
            tf.RandomResizedCrop(size, (0.8, 1)),
            tf.RandomHorizontalFlip(),
            tf.ToTensor(),
            tf.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def valid_tf(size):
    return tf.Compose(
        [
            SquarePad(255),
            tf.Resize(size),
            tf.ToTensor(),
            tf.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
