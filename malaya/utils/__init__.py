from malaya_boilerplate.utils import (
    available_device,
    available_gpu,
    close_session,
)
from malaya_boilerplate import utils
from malaya import package


def print_cache(location=None):
    return utils.print_cache(package=package, location=location)


def delete_cache(location):
    return utils.delete_cache(package=package, location=location)


def delete_all_cache():
    return utils.delete_all_cache(package=package)
