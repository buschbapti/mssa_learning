import socket
import datetime


class Mock(object):
    def __init__(self):
        self.UDP_IP = "0.0.0.0"
        self.UDP_PORT_RCP = 49195
        self.UDP_PORT_EMIT = 49196
        self.SEPARATOR = '@'
        self.SOCK = socket.socket(socket.AF_INET, # Internet
                                  socket.SOCK_DGRAM) # UDP

    def play(self, path):
        with open(path, "r") as rpl_file:
            start = datetime.datetime.now()
            try:
                for line in rpl_file:
                    data, timestamp = line.split(self.SEPARATOR)
                    offset_data = start + datetime.timedelta(seconds=float(timestamp))
                    while "le temps est okay, on peut envoyer la donnee":
                        now = datetime.datetime.now()
                        if now > offset_data:
                            self.SOCK.sendto(data, (self.UDP_IP, self.UDP_PORT_EMIT))
                            break
            except Exception, e:
                print(e)
                print(line)
                raise e

    def record(self, path):
        with open(path, "w") as rpl_file:
            datetime.datetime.now()
            self.SOCK.bind((self.UDP_IP, self.UDP_PORT_RCP))
            start = datetime.datetime.now()
            while True:
                data, addr = self.SOCK.recvfrom(8294400) # buffer size is 1024 bytes
                rpl_file.write(data)
                rpl_file.write(self.SEPARATOR+str((datetime.datetime.now() - start).total_seconds())+'\n')
