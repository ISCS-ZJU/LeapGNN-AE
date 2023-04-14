import netifaces

# 获取本机ipv4和网卡号
def get_local_ip4_and_interface():
    """Return a tuple containing the local IPv4 address and the interface name"""
    ip_interfaces_lst = []
    for interface in netifaces.interfaces():
        if interface.startswith('lo'):
            continue
        addrs_dict = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs_dict:
            for tmp_dict in addrs_dict[netifaces.AF_INET]:
                if 'addr' in tmp_dict:
                    ip_interfaces_lst.append((tmp_dict['addr'], interface))
    return ip_interfaces_lst  # like [('10.214.242.140', 'eno3')]
