# Custom boot — skips stock RC stack, runs Legion firmware
import bbl_product
import sys
import gc

bbl_product.set_app_name("LEGION")
bbl_product.set_app_version("00.01.00.00")
del bbl_product

sys.path.append("/lib")
gc.collect()

import main
main.run()
