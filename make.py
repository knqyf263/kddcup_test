#!/usr/bin/python
#coding:utf-8

import pandas,os,pickle

col_names = ["duration","protocol_type","service","flag","src_bytes",
                "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins", "logged_in","num_compromised","root_shell","su_attempted","num_root",
                "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                "is_host_login","is_guest_login","count","srv_count","serror_rate",
                "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
                "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

#kdd_data = pandas.read_csv("test", header=None, names = col_names)
kdd_data = pandas.read_csv("kddcup99/corrected", header=None, names = col_names)
#kdd_data = pandas.read_csv("kddcup99/kddcup.data_10_percent", header=None, names = col_names)
#kdd_data = pandas.read_csv("kddcup99/kddcup.data_100", header=None, names = col_names)

with open("kddcup99/training_attack_types") as f:
    attack = {"normal": "normal"}
    for e in f:
        k,v = e.strip().split(" ")
        attack[k] = v

kdd_data["label"] = map(lambda x: attack[x.strip("\.")], kdd_data["label"])

obj = {}
def create_dict(col_name):
    l = list(set(kdd_data[col_name]))
    d = {v: i for (i, v) in enumerate(l)}
    obj[col_name] = d

col_list = ['protocol_type', 'service', 'flag', 'label']

if os.path.isfile('mapping.pkl'):
    print "mapping load!"
    with open('mapping.pkl', 'rb') as f:
        obj = pickle.load(f)
else:
    [create_dict(c) for c in col_list]
    with open('mapping.pkl', 'wb') as f:
        pickle.dump(obj, f)

for col_name in col_list:
    kdd_data[col_name] = map(lambda x: obj[col_name][x], kdd_data[col_name])

kdd_data.to_csv('kddcup99/corrected.csv', index=False)
