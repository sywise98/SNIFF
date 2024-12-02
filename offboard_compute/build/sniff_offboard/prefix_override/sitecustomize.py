import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sahaj/SNIFF/offboard_compute/install/sniff_offboard'
