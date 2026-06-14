from unittest.mock import patch
from lab.backends.skypilot import confirm_no_rental


def test_confirm_true_when_no_matching_rental():
    with patch("lab.backends.skypilot.list_vast_instances", return_value=[]):
        assert confirm_no_rental("lab-j-sky") is True


def test_confirm_false_when_rental_still_present():
    inst = [{"id": 1, "label": "lab-j-sky-abc"}]
    with patch("lab.backends.skypilot.list_vast_instances", return_value=inst), \
         patch("lab.backends.skypilot._instance_label", side_effect=lambda i: i["label"]):
        assert confirm_no_rental("lab-j-sky") is False


def test_confirm_false_when_listing_fails():
    with patch("lab.backends.skypilot.list_vast_instances", side_effect=RuntimeError("api down")):
        assert confirm_no_rental("lab-j-sky") is False  # uncertainty -> not confirmed gone
