from typing import List
import random


class NodeInfo:
    '''
    Represents node in lightning network
    '''

    software_probability: dict = None

    def __init__(self, software: str):
        self.software = software

    @classmethod
    def init_software_probability(cls):
        import requests
        res = requests.get('https://ln.bigsun.xyz/api/nodes?select=software')

        softwares = list(map(
            lambda node: node['software'],
            res.json()))

        # count occurences of each software
        lnd = softwares.count('lnd')
        c_lightning = softwares.count('c-lightning')
        eclair = softwares.count('eclair')

        # find total
        total = lnd + c_lightning + eclair

        # set probability for each software type
        NodeInfo.software_probability = dict(
            lnd=lnd/total, c_lightning=c_lightning/total, eclair=eclair/total)

    @classmethod
    def init_random(cls):
        # initialize probabilities for software choice
        if cls.software_probability is None:
            cls.init_software_probability()

        # pick software with weighted probability
        software = random.choices(list(cls.software_probability.keys()),
                                  list(cls.software_probability.values()))

        return NodeInfo(software)


class ChannelInfo:
    '''
    Represents channel in lightning network.
    Values are in Satoshi.
    '''

    # 2 USD (average of past values)
    default_channel_cost = 17000

    default_base_fee = 1
    default_prop_fee_millionth = 1
    default_capacity = 1000

    def __init__(self, capacity: int, base_fee: int, prop_fee_millionth: int, cltv_delta: int):
        self.capacity = capacity
        self.base_fee = base_fee
        self.prop_fee = prop_fee_millionth
        self.cltv_delta = cltv_delta

    @classmethod
    def init_random(cls):
        return ChannelInfo(100000, 1, 1, 144)

    def calc_weight(self, amt: int):
        return self.base_fee + (amt / 1000000) * self.prop_fee
