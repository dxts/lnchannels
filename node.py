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
        cls.software_probability = dict(
            lnd=lnd/total, c_lightning=c_lightning/total, eclair=eclair/total)

    @classmethod
    def init_random(cls):
        # initialize probabilities for software choice
        if cls.software_probability is None:
            cls.init_software_probability()

        # pick software with weighted probability
        software = random.choices(list(cls.software_probability.keys()),
                                  list(cls.software_probability.values()),
                                  k=1)[0]

        return NodeInfo(software)


class ChannelInfo:
    '''
    Represents channel in lightning network.
    Values are in milliSatoshi.
    '''

    # 1.5 USD (average of past values)
    default_channel_cost = 12000000

    policies: dict = None

    def __init__(self, capacity: int, base_fee: int, prop_fee_millionth: int, cltv_delta: int):
        self.capacity = capacity
        self.base_fee = base_fee
        self.prop_fee = prop_fee_millionth
        self.cltv_delta = cltv_delta

    @classmethod
    def init_policies(cls):
        import requests
        from collections import Counter

        res = requests.get(
            'https://ln.bigsun.xyz/api/policies?select=base_fee_millisatoshi,fee_per_millionth,delay')

        # count occurence of each policy
        sampled_policies = Counter(str(p) for p in res.json())

        # top n policies will be sampled
        n = 20

        # total freq of sampled policies
        total = 0
        for p in sampled_policies.most_common(n):
            total += p[1]

        # pick most common policies
        policies = {}
        for p in sampled_policies.most_common(n):
            policies[p[0]] = p[1] / total

        cls.policies = policies

    @classmethod
    def init_random(cls):
        import re
        if cls.policies is None:
            cls.init_policies()

        # pick a policy from the samples
        policy = random.choices(list(cls.policies.keys()),
                                list(cls.policies.values()), k=1)[0]
        match = re.match(
            "{'.*': ([0-9]+), '.*': ([0-9]+), '.*': ([0-9]+)}", policy)

        return ChannelInfo(10000000, float(match[1]), float(match[2]), float(match[3]))

    def calc_fee(self, amt: int):
        """ LND weight function. Fee should be in millisatoshi.
        """
        fee = self.base_fee + (amt / 1000000) * self.prop_fee
        return fee + amt * self.cltv_delta * 15 / 1000000000
