import cv2
import numpy as np

from preprocess import prepare_gray


def test_prepare_gray_matches_simple_training_style_resize():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = 200

    expected = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    expected = cv2.resize(expected, (28, 28), interpolation=cv2.INTER_AREA)

    result = prepare_gray(img, size=(28, 28))

    assert result.shape == (28, 28)
    np.testing.assert_array_equal(result, expected)
