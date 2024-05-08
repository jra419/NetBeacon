import numpy as np


def find_next_split(minz, maxz):
    count = 0

    while (minz >> count) & 1 == 0 and (minz + (1 << count)) < maxz:
        count += 1

    if (minz + ( 1 << count)) > maxz:
        return 1 << (count - 1)

    return 1 << count

# Prefix method
# return value: for example, range_to_ternary(0,48) return [0, 32] and [32, 16],
# Two prefixes, the first covering 32 numbers starting from 0
# and the second covering 16 numbers starting from 32
def range_to_ternary(minz, maxz):
    if maxz <= minz:
        return [[], []]

    start_num = []
    bcount = []

    while True:
        a = find_next_split(minz, maxz)
        start_num.append(minz)
        bcount.append(a)
        if minz + a == maxz:
            break
        minz += a

    return start_num, bcount

# length: the number of mask bits
# num: the number of covers, i.e., the bcount returned by range_to_ternary
def get_mask(length, num):
    a = int(np.log2(num))
    result = '0b'

    for i in range(length - a):
        result += '1'
    # Mask position is 0
    for i in range(a):
        result += '0'

    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Convert 1**00 to 10000 and mask 10011
# bits is the number of bits
def get_value_mask(s, bits):
    value = s.copy()
    mask = ['1'] * bits

    for i in range(len(s)):
        if s[i] == '*':
            value[i] = '0'
            # 0 is anything
            mask[i] = '0'

    return ''.join(value), ''.join(mask)

# Get the range mark of the basis range in the feature table
# length: range mark bits
# valid: the number of feature thresholds
# num: the serial number of basis range
def get_feature_table_range_mark(length, num, valid):
    result = '0b'
    for i in range(num):
        result += '0'
    for i in range(valid - num):
        result += '1'
    # Invalid bits default to 0
    for i in range(length - valid):
        result += '0'
    return result

# Get the range mark of the associate range in the model table
# length: the bits of range mark
# valid: the number of feature thresholds
# a,b: the left and right of associate range
def get_model_table_range_mark(length, a, b, valid):
    result = ['0'] * length
    for n in range(valid):
        if n < abs(b):
            result[n] ='0'
        elif n >= a - 1:
            result[n] = '1'
        else:
            result[n] = '*'
    return result

trans = lambda x: list(map(float, x.strip('[').strip(']').split(',')))

#Turn the value in the rf leaf node into a probability
def list_to_proba(ll):
    ll = trans(ll)
    re = []
    for i in ll:
        re.append(i / np.sum(ll))
    return re

# Get the feature table entries, including the key and action parameters
# feat_dict: the threshold value of each feature
# key_bits: the number of bits of the feature
# key_encode_bits: the number of bits of the range mark
# pkts: the number of pkts packets, optional
# return value is a dict, key is feature, value is list
# each item in the list is a table item in the corresponding feature table, including:
# [priority, value, mask, action parameter, (number of packets)
def get_feature_table_entries(feat_dict, key_bits, key_encode_bits, pkts=None):
    feat_table_datas = {}

    for key in feat_dict.keys():
        thres = feat_dict[key]
        feat_table = []
        sum1 = 0
        priority = 1

        for i in range(len(thres)):
            best_start = 0
            best_end = 0
            min_entries = 100000
            end = thres[i]

            while (i != len(thres) - 1 and end < thres[i+1]) \
                    or (i == len(thres) - 1 and end < (2**key_bits[key])):
                right_entries = len(range_to_ternary(thres[i], end)[0])
                start = thres[i-1] if i > 0 else 0

                while start >= 0:
                    temp = range_to_ternary(start, end)
                    now_entries = len(temp[0]) + right_entries
                    if now_entries < min_entries:
                        min_entries = now_entries
                        best_start = start
                        best_end = end
                    if len(temp[0]) > 1 and temp[1][0] - temp[1][1] < 0:
                        start += temp[1][0] - temp[1][1]
                    else:
                        break

                if len(temp[0]) > 1 and temp[1][-1] - temp[1][-2] < 0:
                    end += temp[1][-2] - temp[1][-1]
                else:
                    break

            #print(min_entries, thres[i-1], thres[i], best_start, best_end)
            temp = range_to_ternary(thres[i], best_end)
            for j in range(len(temp[0])):
                feat_table.append([priority,temp[0][j],
                                   int(get_mask(key_bits[key], temp[1][j]), 2),int(get_feature_table_range_mark(key_encode_bits[key], i+1,
                                                                    len(thres)), 2)])
                if pkts is not None:
                    feat_table[-1].append(pkts)

            priority += 1
            temp = range_to_ternary(best_start, best_end)
            for j in range(len(temp[0])):
                feat_table.append([priority,temp[0][j],
                                   int(get_mask(key_bits[key],temp[1][j]),2),
                                   int(get_feature_table_range_mark(key_encode_bits[key], i,
                                                                    len(thres)), 2)])
                if pkts is not None:
                    feat_table[-1].append(pkts)

            priority += 1
            sum1 += min_entries

        print("The entries of {} is {}.".format(key, len(feat_table)))
        feat_table_datas[key] = feat_table

    return feat_table_datas

# Convert aggregate feature to table
def get_bin_table(keys, bin_count_bits, QL=4):
    bin_table_data = []
    mask_value = (((2**bin_count_bits)-1) >> QL) << QL

    for key in keys:
        if "bin" in key:
            start_value = int(key[4:]) * 2 ** QL
            bin_table_data.append([start_value, mask_value])

    return bin_table_data
