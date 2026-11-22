"""StockMind: PSX stock analysis and decision support."""

import logging


# psxdata logs routine warnings on every download. Prices are sorted and
# validated afterwards (see `price_issues`), and StockMind's own warnings are
# attached to each result and printed with it, so only errors are logged.
logging.getLogger("psxdata").setLevel(logging.ERROR)
logging.getLogger("stockmind").setLevel(logging.ERROR)
