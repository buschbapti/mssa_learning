import datetime
import requests


class CommandKalman(object):
    def __init__(self):
        self.HOST = "http://127.0.0.1:9001"
        self.SESSION = {'worker': 'Automated',
                        'observator': 'observator_test',
                        'customer': 'customer_test',
                        'workstation': 'workstation_test'}

    def call(self, method, url, headers=None, **kwargs):
        req_headers = dict()
        if headers:
            req_headers.update(headers)
        return requests.request(method=method, url=self.HOST + url, headers=req_headers, **kwargs)


    def is_status_ok(self, status):
        return 200 <= status < 300


    def start_record(self):
        return self.call('POST', url='/record/start', json=self.SESSION)


    def stop_record(self):
        return self.call('POST', url='/record/stop', json={'observator': 'observator_test'})
