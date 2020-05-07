
## Protocol Mappings
protocol = ['icmp', 'tcp', 'udp']

## Service Mappings
service = [ 
            'other', 'link', 'netbios_ssn', 'smtp', 'netstat', 'ctf', 'ntp_u', 'harvest', 'efs', 'klogin', 'systat', 
            'exec', 'nntp', 'pop_3', 'printer', 'vmnet', 'netbios_ns', 'urh_i', 'ssh', 'http_8001', 'iso_tsap', 'aol',
            'sql_net', 'shell', 'supdup', 'auth', 'whois', 'discard', 'sunrpc', 'urp_i', 'rje', 'ftp', 'daytime',
            'domain_u', 'pm_dump', 'time', 'hostnames', 'name', 'ecr_i', 'bgp', 'telnet', 'domain', 'ftp_data', 'nnsp',
            'courier', 'finger', 'uucp_path', 'X11', 'imap4', 'mtp', 'login', 'tftp_u', 'kshell', 'private', 'http_2784',
            'echo', 'http', 'ldap', 'tim_i', 'netbios_dgm', 'uucp', 'eco_i', 'remote_job', 'IRC', 'http_443', 'red_i',
            'Z39_50', 'pop_2', 'gopher', 'csnet_ns'
          ]

## Flag Mappings
flag = ['OTH', 'S0', 'S1', 'S2', 'S3', 'RSTO', 'REJ', 'SF', 'SH', 'RSTR', 'RSTO0', 'RSTOS0']

## Attack Mappings
attacks = dict({ "normal": 0 })
attacks.update(dict.fromkeys([ "apache2", "back", "land", "neptune", "mailbomb", "pod", "processtable", "smurf", "teardrop", "udpstorm", "worm" ], 1))
attacks.update(dict.fromkeys([ "ipsweep", "mscan", "nmap", "portsweep", "saint", "satan" ], 2))
attacks.update(dict.fromkeys([ "buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm" ], 3))
attacks.update(dict.fromkeys([ "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf", "sendmail", "snmpgetattack", "spy", "snmpguess", "warezclient", "warezmaster", "xlock", "xsnoop" ], 4))
