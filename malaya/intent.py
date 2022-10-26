from malaya.function import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {

}


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info('tested on ms-MY MASSIVE test set, https://huggingface.co/datasets/qanastek/MASSIVE')
    return describe_availability(_huggingface_availability)
