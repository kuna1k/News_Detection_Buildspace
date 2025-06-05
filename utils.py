import logging

def setup_logging():
    """
    Call this at the top of train.py or predict.py to set logger format & level. 
    This helps to see what is actually happening in real time, example you see an error you can see more clearly where.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
